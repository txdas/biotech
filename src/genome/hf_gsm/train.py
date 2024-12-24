from dataset import DNAData,collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import GlmConfig
from model_base import GlmForMaskedLM


if __name__ == '__main__':
    dataloader = DataLoader(dataset=DNAData(), collate_fn=collate_fn, batch_size=16)
    from transformers import TrainingArguments,Trainer
    conf = GlmConfig().from_pretrained("config")
    model = GlmForMaskedLM(conf)
    training_args = TrainingArguments(output_dir="test_trainer",logging_steps=8,)
    ds = DNAData()
    print(len(ds[0]['input_ids']),ds[0]['input_ids'][412])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,data_collator= collate_fn
    )
    trainer.train()
    # for v in tqdm(dataloader):
    #     pass
        # print(v["input_ids"].shape)

