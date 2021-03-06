from fairseq.models.bart import BARTModel
from typing import Dict
import json
from copy import deepcopy
import argparse


def fix_tokenizer(old_tokenizer, new_vocab: Dict[str, int]):
    """
    The new_tokenizer is a copy of the old_tokenizer and is supplied instead the
    new vocabulary dictionary.
    Since the keys must match the merges, we append keys from the old_tokenizer
    to the new_tokenizer.
    """
    new_tokenizer = deepcopy(old_tokenizer)
    new_tokenizer["model"]["vocab"] = deepcopy(new_vocab)
    for k in old_tokenizer["model"]["vocab"]:
        if k not in new_vocab:
            new_tokenizer["model"]["vocab"][k] = len(new_tokenizer["model"]["vocab"])

    # Change <mask> token id from 4 to 50184
    new_tokenizer["added_tokens"][4]["id"] = 50184
    return new_tokenizer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer.json",
        help="Path to the huggingface tokenizer-json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_best.pt",
        help="Name of the BART checkpoint file.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="bart_model",
        help="Path to the folder containing a BART checkpoint and dict.txt",
    )
    parser.add_argument(
        "--new_tokenizer",
        type=str,
        default="tokenizer_fixed.json",
        help="Path to the new tokenizer-json",
    )
    return parser.parse_args()


def main():

    args = get_args()

    with open(args.tokenizer) as fin:
        tokenizer = json.load(fin)

    bart = BARTModel.from_pretrained(args.folder, checkpoint_file=args.checkpoint)
    model_dict = bart.task.source_dictionary.__dict__
    new_vocab = model_dict["indices"]

    fixed_tokenizer = fix_tokenizer(tokenizer, new_vocab)

    with open(args.new_tokenizer, "w") as fout:
        json.dump(fixed_tokenizer, fout, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
