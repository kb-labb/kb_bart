import torch
from torch import nn
import transformers
import argparse


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_best.pt")
    return parser.parse_args()


def main():
    args = get_args()

    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        cls_token="</s>",
        sep_token="</s>",
    )

    state_dict = torch.load(args.checkpoint, map_location="cpu")["model"]

    vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]

    config = transformers.BartConfig(
        vocab_size=vocab_size,
        d_model=768,
        decoder_ffn_dim=3072,
        encoder_ffn_dim=3072,
        decoder_layers=6,
        encoder_layers=6,
    )

    model = transformers.BartForConditionalGeneration(config)

    remove_ignore_keys_(state_dict)
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    model.model.load_state_dict(state_dict)

    model.lm_head = make_linear_from_emb(model.model.shared)

    # Save to Huggingface format
    model.save_pretrained("hfmodel")
    tok.save_pretrained("hfmodel")


if __name__ == "__main__":
    main()
