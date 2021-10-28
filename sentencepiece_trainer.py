import sentencepiece as spm
from datasets import load_dataset

dataset = load_dataset("oscar", "unshuffled_deduplicated_sv")


def batch_iterator(dataset, dataset_size, batch_size):
    for i in range(0, dataset_size, batch_size):
        # Tokenizers ignore new lines, but when writing to .txt-file
        # we don't want newlines inserted at every \n, only at the end
        text_batch = map(
            lambda text: text.replace("\n", " ") + "\n", dataset[i : i + batch_size]["text"]
        )
        yield list(text_batch)


def create_txt_from_dataset(dataset, text_line_generator):
    with open("output.txt", "w") as f:
        for line in text_line_generator:
            f.writelines(line)


text_line_generator = batch_iterator(dataset["train"], 11000000, 50)
create_txt_from_dataset(dataset["train"], text_line_generator)

spm.SentencePieceTrainer.train(
    input="output.txt",
    model_prefix="spm.bpe",
    vocab_size=50260,
    user_defined_symbols=["<mask>"],
    model_type="bpe",
)

