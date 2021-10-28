from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
    processors,
)

dataset = load_dataset("oscar", "unshuffled_deduplicated_sv")


def batch_iterator(dataset, dataset_size, batch_size):
    for i in range(0, dataset_size, batch_size):
        # Tokenizers ignore new lines, but when writing to .txt-file
        # we don't want newlines inserted at every \n, only at the end
        text_batch = map(
            lambda text: text.replace("\n", " ") + "\n", dataset[i : i + batch_size]["text"]
        )
        yield list(text_batch)


# https://github.com/huggingface/tokenizers/issues/640#issuecomment-792305076
def bpe_tokenizer_trainer(text, vocab_size, min_frequency=0):
    # Supply either path to txt file or list of strings as text arg

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        min_frequency=min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    if isinstance(text, str):
        # if user specified path to txt file as string
        tokenizer.train(text, trainer=trainer)
    else:
        tokenizer.train_from_iterator(text, trainer=trainer)

    tokenizer.post_processor = processors.RobertaProcessing(
        sep=("</s>", tokenizer.token_to_id("</s>")), cls=("<s>", tokenizer.token_to_id("<s>"))
    )

    tokenizer.save("tokenizer.json")
    # tokenizer.model.save("output_dir")


def pretokenizer_print(text):
    # To print and check how pre tokenization looks like
    pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    return pre_tokenizer.pre_tokenize_str(text)


pretokenizer_print(dataset["train"][0]["text"])

bpe_tokenizer_trainer(text=dataset["train"][0:3100000]["text"], vocab_size=50260)
