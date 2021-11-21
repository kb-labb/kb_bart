import os
import argparse
import time
import multiprocessing as mp
import fairseq
from transformers import PreTrainedTokenizerFast

# Run this file on slurm via tokenizers_encoder_run.sh

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", type=str)
parser.add_argument(
    "data_folder",
    nargs="?",
    type=str,
    default="/ceph/hpc/home/eufatonr/data/text/kb_bart_data/split",
)
parser.add_argument(
    "-d",
    "--dest_folder",
    nargs="?",
    type=str,
    default="/ceph/hpc/home/eufatonr/data/text/kb_bart_data/tokenized",
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


def per_document(iterator, is_delimiter=lambda x: x.isspace()):
    """
    # Read text file where sentences are separated by newline and 
    # documents by empty line into a list of lists.
    # https://stackoverflow.com/questions/25226871/splitting-textfile-into-section-with-special-delimiter-line-python/25226944#25226944
    """
    sentences = []
    for line in iterator:
        if is_delimiter(line):
            if sentences:
                yield sentences  # OR  ''.join(sentences)
                sentences = []
        else:
            sentences.append(line.rstrip())  # OR  sentences.append(line)
    if sentences:
        yield sentences


def tokenize_text(document):
    """
    Document is a list of lists where each nested list is a sentence.
    [[sentence1], [sentence2], ...]
    """
    tokenized_sentences = []
    for sentence in document:
        tokenized_sentence = tokenizer.tokenize(sentence)
        tokenized_sentence = " ".join(tokenized_sentence)
        tokenized_sentences.append(tokenized_sentence)

    return tokenized_sentences


def split_long_docs(doc, max_len=1022):
    """
    Split documents longer than 1022 tokens into chunks
    """
    new_doc = []
    doc_len = 0
    for i, sen in enumerate(doc):
        sen_len = len(sen.split())  # word split
        if doc_len + sen_len < max_len:
            new_doc.append(sen)
            doc_len += sen_len
        else:
            yield new_doc
            new_doc = [sen]
            doc_len = sen_len
    yield new_doc


def preprocess_text(document, max_sequence_length=1022):
    tokenized_document = tokenize_text(document)
    total_doc_length = sum([len(sentence) for sentence in tokenized_document])
    if total_doc_length > max_sequence_length:
        tokenized_document_splits = split_long_docs(tokenized_document, max_sequence_length)
        return list(tokenized_document_splits)
    else:
        return [tokenized_document]


# data_folder = "/ceph/hpc/home/eufatonr/data/text/kb_bart_data/split"
text_shard_file = os.path.join(args.data_folder, args.filename)

t0 = time.time()
with open(text_shard_file) as f:
    documents = list(per_document(f))  # default delimiter
t1 = time.time()
print(f"Reading sentences from file {text_shard_file}. Completed in {t1 - t0} seconds.")

t0 = time.time()
pool = mp.Pool(processes=20)
tokenized_sentences = pool.map(preprocess_text, documents)
pool.close()
t1 = time.time()

# Unnest the inner lists in tokenized_sentences
flat_list = [item for sublist in tokenized_sentences for item in sublist]
flat_list = [" ".join(sen) for sen in flat_list]  # join list of sentences to doc

output_filename = os.path.basename(text_shard_file) + ".docs.token"
output_path = os.path.join(args.dest_folder, output_filename)

# Use regular file write line by line to avoid quoting/escaping issues of pandas.
with open(output_path, "w") as wf:
    for line in flat_list:
        wf.write(line)
        wf.write("\n")


print(f"{os.path.basename(text_shard_file)} has been tokenized and saved to {output_path}.")
print(f"Time to tokenize sentences in shard: {t1 - t0} seconds.")
