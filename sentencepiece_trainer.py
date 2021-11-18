import sentencepiece as spm

# from datasets import load_dataset

# dataset = load_dataset("oscar", "unshuffled_deduplicated_sv", cache_dir="/ceph/hpc/home/eufatonr/faton/kb_bart/oscar")


def batch_iterator(dataset, dataset_size, batch_size):
    for i in range(0, dataset_size, batch_size):
        # Tokenizers ignore new lines, but when writing to .txt-file
        # we don't want newlines inserted at every \n, only at the end
        text_batch = map(
            lambda text: text.replace("\n", " ") + "\n", dataset[i : i + batch_size]["text"]
        )
        yield list(text_batch)


def create_txt_from_dataset(text_line_generator, filename):
    with open(filename, "w") as f:
        for line in text_line_generator:
            f.writelines(line)


# text_line_generator = batch_iterator(dataset["train"], len(dataset["train"]), 50)
# create_txt_from_dataset(text_line_generator, "oscar_train.txt")

spm.SentencePieceTrainer.train(
    input="oscar_train.txt",
    model_prefix="spm.bpe",
    vocab_size=50265,
    user_defined_symbols=["<mask>"],
    model_type="bpe",
    bos_id=0,
    pad_id=1,
    eos_id=2,
    unk_id=3,
)
