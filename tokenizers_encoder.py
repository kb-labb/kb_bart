import os
import argparse
import time
import pandas as pd
import csv
import multiprocessing as mp
from transformers import PreTrainedTokenizerFast

# Run this file on slurm via tokenizers_encoder_run.sh

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", type=str)
parser.add_argument(
    "data_folder", nargs="?", type=str, default="/ceph/hpc/home/eufatonr/data/text/kb_bart_data"
)
args = parser.parse_args()

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer.json",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    mask_token="<mask>",
    pad_token="<pad>",
)

data_folder = "/ceph/hpc/home/eufatonr/data/text/kb_bart_data"
text_shard_file = os.path.join(args.data_folder, args.filename)


def tokenize_text(sentence):
    tokenized_sentence = tokenizer.tokenize(sentence)
    tokenized_sentence = " ".join(tokenized_sentence)
    return tokenized_sentence


t0 = time.time()
df = pd.read_csv(text_shard_file, sep="\\n", header=None, engine="python")
df = df.rename(columns={0: "text"})
t1 = time.time()
print(f"pd.read_csv() of file {text_shard_file} completed in {t1 - t0} seconds.")


t0 = time.time()
pool = mp.Pool(processes=20)
tokenized_sentences = pool.map(tokenize_text, df["text"].tolist())
pool.close()
t1 = time.time()

os.makedirs(os.path.join(data_folder, "tokenized"), exist_ok=True)

output_filename = os.path.basename(text_shard_file) + ".token"
output_path = os.path.join(data_folder, "tokenized", output_filename)
df["tokenized"] = tokenized_sentences

# Use regular file write line by line to avoid quoting/escaping issues of pandas.
with open(output_path, "w") as wf:
    for i, line in df["tokenized"].iteritems():
        wf.write(line)
        wf.write("\n")

print(f"{os.path.basename(text_shard_file)} has been tokenized and saved to {output_path}.")
print(f"Time to tokenize sentences in shard: {t1 - t0} seconds.")
