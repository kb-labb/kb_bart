import os
import time
import argparse
import fairseq
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", type=str)
parser.add_argument("-w", "--num_workers", type=int)
parser.add_argument("--dictionary", type=str, default="dict.txt")
parser.add_argument(
    "data_folder",
    nargs="?",
    type=str,
    default="/ceph/hpc/home/eufatonr/data/text/kb_bart_data/tokenized",
)
parser.add_argument(
    "-d",
    "--dest_folder",
    nargs="?",
    type=str,
    default="/ceph/hpc/home/eufatonr/data/text/kb_bart_data/tokenized",
)
args = parser.parse_args()

d = fairseq.data.Dictionary.load(args.dictionary)


def chunks(l, n):
    """Yield n number of striped chunks from l.
    https://stackoverflow.com/questions/24483182/python-split-list-into-n-chunks/48971420
    """
    for i in range(0, n):
        yield l[i::n]


def validate_sentences(doc_chunk):
    """
    Check if <s> occurs naturally in the middle of a document.
    Remove the observation, otherwise training will crash with
    an assertionerror that (sentence[1:-1] >= 1).all().
    <s> is encoded as 0 in the dictionary/vocabulary.
    """
    output_docs = []
    for doc in doc_chunk:
        encoded_sen = d.encode_line(doc, add_if_not_exist=False)

        if (encoded_sen[1:-1] >= 1).all():
            output_docs.append(doc)
        else:
            print(f"Removing observation, document: {doc}")
    return output_docs


text_shard_file = os.path.join(args.data_folder, args.filename)

documents = []
with open(text_shard_file) as f:
    for line in f:
        documents.append(line)

doc_chunks = list(chunks(documents, args.num_workers))

t0 = time.time()
pool = mp.Pool(processes=args.num_workers)
validated_sentences = pool.map(validate_sentences, doc_chunks)
t1 = time.time()
print(f"Documents in file {text_shard_file} validated in {t1 - t0} seconds.")


flat_list = [item for sublist in validated_sentences for item in sublist]

output_filename = os.path.basename(text_shard_file) + ".check"
output_path = os.path.join(args.dest_folder, output_filename)

with open(output_path, "w") as wf:
    for line in flat_list:
        wf.write(line)
