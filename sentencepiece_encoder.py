import sentencepiece as spm

# Best let dataset/dataloader insert bos and eos rather than do it here
sp = spm.SentencePieceProcessor(model_file="spm.bpe.model",)  # add_bos=True, add_eos=True

sp.encode("Det här är en överdriven testmening", out_type=str)
sp.encode("<s> Testa vad som sker med BOS- och EOS-symboler</s>", out_type=str)


with open("oscar_train.txt", "r") as rf, open("oscar_train.bpe", "w") as wf:
    for line in rf:
        wf.write(" ".join(sp.encode(line, out_type=str)))
        wf.write("\n")

with open("oscar_valid.txt", "r") as rf, open("oscar_valid.bpe", "w") as wf:
    for line in rf:
        wf.write(" ".join(sp.encode(line, out_type=str)))
        wf.write("\n")

print("Done.")