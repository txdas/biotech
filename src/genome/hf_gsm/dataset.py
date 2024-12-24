from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import random
from Bio import SeqIO
from tokenizer import GlmTokenizer
pvf = "C:\\Users\\jinya\\Desktop\\bio\\data\\pa"
tokenize = GlmTokenizer(vocab_file="./config/vocab.txt")


def load_data():
    random.seed(41)
    min_len = 50
    buffer, step, slen = [],510, 510
    with open(f"{pvf}\\apv.fasta") as input_handle:
        for record in SeqIO.parse(input_handle, "fasta"):
            seq = str(record.seq)
            for i in range(step,len(seq),step):
                sseq = seq[i:i+slen]
                if len(sseq)<min_len:
                    continue
                yield mask_seq(sseq)
                # if buffer and random.random()>0.5:
                #     rseq = random.choice(buffer)
                #     yield mask_seq(sseq, rseq, 0)
                # else:
                #     if i+slen<len(seq)-min_len:
                #         rseq = seq[i+slen:i+2*slen]
                #         yield mask_seq(sseq, rseq, 1)
                # buffer.append(sseq)

def mask_seq(sseq):
    def mask(c):
        ch = c
        if random.random() < 0.15:
            rand = random.random()
            if rand < 0.8:
                ch = "<mask>"
            elif rand < 0.9:
                chs = [v for v in "ACGT" if v != ch]
                ch = random.choice(chs)
        return ch

    msseq = "".join([ mask(c) for c in sseq])
    return msseq, sseq


class DNAData(Dataset):
    def __init__(self):
        super().__init__()
        self.seqs = [v for v in load_data()][:400]
        self.tokenizer = tokenize

    def __getitem__(self, idx):
        seq, target = self.seqs[idx]
        seq_output = self.tokenizer(seq)
        target_output = self.tokenizer(target)
        return {"input_ids":seq_output["input_ids"],"attention_mask":seq_output["attention_mask"],
                "labels":target_output["input_ids"]}

    def __len__(self):
        return len(self.seqs)


def collate_fn(batch):
    datas = [v for v in batch]
    max_len = max([ len(v["input_ids"]) for v in datas])
    for v in datas:
        pad_len = max_len - (len(v["input_ids"]))
        if pad_len>0:
            v["input_ids"] = v["input_ids"]+[tokenize.pad_token_type_id]*pad_len
            v["attention_mask"] = v["attention_mask"] + [0]*pad_len
            v["labels"] = v["labels"]+[tokenize.pad_token_type_id]*pad_len

    return {"input_ids": torch.tensor([v["input_ids"]for v in datas]),
            "attention_mask": torch.tensor([v["attention_mask"] for v in datas]),
            "labels": torch.tensor([v["labels"] for v in datas]),
            }


if __name__ == '__main__':
    ds = DNAData()
    print(ds[0])
    dataloader = DataLoader(dataset=ds,batch_size=4,collate_fn=collate_fn)
    for v in dataloader:
        print(v["input_ids"].shape)
        break


