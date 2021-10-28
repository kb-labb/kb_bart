import sentencepiece as spm

# Best let dataset/dataloader insert bos and eos rather than do it here
sp = spm.SentencePieceProcessor(model_file="spm.bpe.model",)  # add_bos=True, add_eos=True

sp.encode("Det här är en överdriven testmening", out_type=str)
sp.encode("<s> Testa vad som sker med BOS- och EOS-symboler</s>", out_type=str)

with open("output00", "r") as rf, open("file00.bpe", "w") as wf:
    for line in rf:
        wf.write(" ".join(sp.encode(line, out_type=str)))
        wf.write("\n")
