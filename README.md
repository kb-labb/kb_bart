## BART pretraining scripts

User should

1. Clone this repo
2. Clone and install the following fork of fairseq [Lauler/fairseq](https://github.com/Lauler/fairseq).
3. (Optional) For faster training, clone and build NVIDIAS's apex library (see instructions above in the fairseq documentation). Make sure that system CUDA version matches the CUDA version pytorch was built with in local python environment before attempting to install apex (otherwise there will be version mismatch).

Optionally users may build a Singularity container that has all these dependencies. See the file `singularity_pytorch_bart.def` for a definition file of the singularity container we built and used ourselves. You can build the same container using the command

```
sudo singularity build pytorch_21.10_bart.sif singularity_pytorch_bart.def
```

This will create a singularity image file `pytorch_21.10_bart.sif` which you can either use interactively via `singularity shell --nv pytorch_21.10_bart.sif` or use to execute scripts via `singularity exec --nv pytorch_21.10_bart.sif ...`, where `...` are programs/commands you want to run or issue.

## Data and tokenization

Assuming you have a large corpus to pretrain on, you should arrange the data into one or several text files with one sequence (can be composed of many sentences) per line. If you have limited GPU memory, consider sharding the text file into multiple chunks. A convenience script for doing this can be found in `shard_txt_file.sh`. 

Two methods for training GPT2-style BPE tokenizers are provided: `tokenizers_trainer.py` (Huggingface's tokenizers) and `sentencepiece_trainer.py` (Google's SentecePiece). 

### Sentencepiece
To use [sentencepiece](https://github.com/google/sentencepiece) first install

```
pip install sentencepiece
```

Then 

1. Run `sentencepiece_trainer.py` to train your tokenizer. This will generate two files: `spm.bpe.model` and `spm.bpe.vocab`. 
2. (Optional) Shard your dataset with `shard_txt_file.sh`.
3. Apply the tokenizer you have trained on the text file shard by running `sentencepiece_encoder.py`. This script will byte pair encode your text and generate output files ending with suffix `.bpe`.
4. Before `fairseq-preprocess` can be applied, we need to edit the `spm.bpe.vocab` file to change the column separator from tab (`\t`) to a space separator, because fairseq expects a space separated vocab file. See [this](https://github.com/musixmatchresearch/umberto/issues/2#issuecomment-585894712) Github thread explaining the process of using a sentencepiece vocab in fairseq. And see [this](https://github.com/facebookresearch/fairseq/issues/1490#issuecomment-566604192) Github comment for context on what information the columns contain. Column nr 2 is the frequency of the token in your training set, but can be set to any dummy integer.

### Huggingface tokenizers
To use tokenizers from Huggingface you

1. Run tokenizers_trainer.py to train your tokenizer. This will generate a file called `tokenizer.json`.
2. (Optional) Shard your datasets using `shard_txt_file.sh`.
3. Apply the tokenizer you have trained to tokenize the text file shards by running `tokenizers_encoder.py`. We launch multiple parallel jobs on SLURM to encode each individual shard via the script `tokenizers_encoder_run.sh`. 

## Preprocess data with fairseq

Once we have tokenized our data and converted it to byte pair encoded format, we are ready to preprocess it to a format `fairseq` can ingest during training. For `sentencepiece` vocab preprocessing, please see this github [comment and issue thread](https://github.com/musixmatchresearch/umberto/issues/2#issuecomment-585894712). If you want to use a Huggingface tokenizer (like we did), we will outline two different ways of doing this.

### Option 1 (untested but should work and be the easiest)

Convert your `tokenizer.json` from `json` format to a `txt` format. Fairseq expects the vocab file to have two columns, and to use a **space separator** between the columns. The first columns is the tokens, and the second column a frequency count of how often the tokens appear in the training dataset. We can insert [dummy placeholder integers instead of actual frequencies](https://github.com/facebookresearch/fairseq/issues/1490#issuecomment-566604192). The converted `dict.txt` file might look something like this:

```
Ġ. 12345
Ġ, 12345
Ġoch 12345
Ġi 12345
Ġatt 12345
ĠÃ¤r 12345
Ġsom 12345
...
```

You will generally see a lot of `Ġ` characters. This is because it is the encoding for a space character in BPE. The number `12345` are simply dummy freqency counts added to the second column because fairseq expects them.

### Option 2 (tested, but we recommend against this option if you manually added tokens to your vocab when training it with Huggingface)

A second option is to let `fairseq-preprocess` automatically generate the `dict.txt` file for us. This can be done by [not specifying the `--srcdict` option](https://github.com/facebookresearch/fairseq/issues/1186#issuecomment-535606529) when running `fairseq-preprocess`.

**IMPORTANT:** You need to run `fairseq-preprocess` once on your entire dataset to generate a `dict.txt` which covers all the text in your training/validation data. If you have sharded your dataset in the previous step, this means you need to (re)create a big text file that combines all the tokenized shards into one giant file. For example something like:

```bash
singularity exec pytorch_21.03_bart.sif \
    fairseq-preprocess --only-source \
    --trainpref "/path/to/all_data.txt" \
    --validpref "/path/to/valid.txt" \
    --destdir "/path/to/dest_folder/all" \
    --workers 64
```

In the above example, the resulting `dict.txt` will be available in the folder `/path/to/dest_folder/all`. A big file of 101 GB took us about 2h 30min to preprocess with 20 workers (threads). With 128 workers the same file was preprocessed in 42 minutes. 

**WARNING:** If you manually enforced the inclusion of certain tokens during the Huggingface tokenizers training, there may be a token mismatch between your original `tokenizer.json` and the `dict.txt` generated by `fairseq-preprocess`. This is possible to fix afterwards by creating a new `tokenizer.json` from the `dict.txt`. But it can be a huge headache, because `tokenizers` in Huggingface is sensitive to the ordering of tokens. Thus we strongly recommend you only use **Option 2** if the set of tokens in your `tokenizer.json` are exactly the same as the set of tokens in `dict.txt`.

### Final preprocessing step when we finally have a `dict.txt`

Once you have a `dict.txt` which covers your entire dataset, you can preprocess the individual shards by pointing to your "master dictionary file" using the `--srcdict` argument in `fairseq-preprocess`. 

```bash
singularity exec pytorch_21.03_bart.sif \
    fairseq-preprocess --only-source \
    --trainpref "/path/to/shard1.txt" \
    --validpref "/path/to/valid.txt" \
    --destdir "/path/to/dest_folder/shard1" \
    --srcdict "/path/to/dict.txt" \
    --workers 64
```

We have a bash script for launching multiple jobs to preprocess the shards in `preprocess.sh`. 

## Train the model

Finally we have everything that is needed to start training. See `train_bart.sh` and `train_bart_args.sh` for a BART pre-training script. 

**IMPORTANT:** Fairseq training will always crash once you run out of shards. It does not allow you to relist the same shard files in order to continue training another epoch on the same files. This behavior might be by design, as it is better to create a fresh shuffling of the data for each subsequent epoch. It is possible to reuse the existing shards by restarting the training. However, we rather recommend creating more shuffled shards of the same data (for however many epochs you expect to train) before starting training.

I have done my best to translate what was written in the paper to fairseq config commands by reading the fairseq docs and the relevant source code. Details on dropout and learning rate are not  clear from paper. You need to set these yourselves and adjust based on your batch size. 

Kindly open an issue if you notice anything strange with my suggested config.
