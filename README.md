## BART pretraining scripts

User should

1. Clone this repo
2. Clone and install [fairseq](https://github.com/pytorch/fairseq#requirements-and-installation).
3. (Optional) For faster training, clone and build NVIDIAS's apex library (see instructions above in the fairseq documentation). Make sure that system CUDA version matches the CUDA version pytorch was built with in local python environment before attempting to install apex (otherwise there will be version mismatch).

Installation and training were tested on Pytorch 1.9.1 with CUDA 11.1, and [compatible cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) (v8.0.5).

## Data and tokenization

Assuming you have a large corpus to pretrain on, you should arrange the data into one or several text files with one sentence/paragraph/sequence per line. If you have limited RAM memory, consider sharding the text file into multiple chunks. A convenience script for doing this can be found in `shard_txt_file.sh`. 

Two methods for training GPT2-style BPE tokenizers are provided: `tokenizers_trainser.py` and `sentencepiece_trainer.py`. We have used -- and recommend using -- `sentencepiece_trainer.py`. To use [sentencepiece](https://github.com/google/sentencepiece) first install

```
pip install sentencepiece
```

Then 

1. Run `sentencepiece_trainer.py` to train your tokenizer. This will generate two files: `spm.bpe.model` and `spm.bpe.vocab`. Train the tokenizer on the full dataset.
2. (Optional) Shard your dataset with `shard_txt_file.sh`.
3. Apply the tokenizer you have trained on the text file shard by running `sentencepiece_encoder.py`. This script will byte pair encode your text and generate output files ending with suffix `.bpe`.

## Preprocess data with fairseq

Once we have tokenized our data and converted it to byte pair encoded format, we are ready to preprocess it to a format `fairseq` can ingest during training. In order to do this run `preprocess.sh`. This will convert the bpe-files to a format that is fast to read from disk (.bin together with a corresponding .idx). The preprocessing script will also generate a dict.txt for us. 

## Train the model

Finally we have everything that is needed to start training. See `train_bart.sh` for a BART training script. 

I have done my best to translate what was written in the paper to fairseq config commands by reading the fairseq docs and the relevant source code. Details on dropout and learning rate are not entirely clear from paper, so I recommend experimenting for yourselves. 

Kindly open an issue if you notice something strange with my suggested config.
