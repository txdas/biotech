from torchinfo import summary
import torch
from bgr.bhi.final_layers_block import BHIFinalLayersBlock
from bgr.bhi.first_layers_block import BHIFirstLayersBlock
from bgr.bhi.coreblock import BHICoreBlock
from bgr.bhi.dataprocessor import BHIDataProcessor
from bgr.models import PrixFixeNet
from bgr.bhi.trainer import BHITrainer
from bgr.bhi.predictor import BHIPredictor
fdir,name = "/Users/john/data/Promter/results", "core6-merge_core"
generator = torch.Generator()
generator.manual_seed(2147483647)
TRAIN_DATA_PATH = f"{fdir}/{name}/{name}_train.csv" #change filename to actual training data
VALID_DATA_PATH = f"{fdir}/{name}/{name}_test.csv" #change filename to actual validaiton data
TRAIN_BATCH_SIZE = 256 # replace with 1024, if 1024 doesn't fit in gpu memory, decrease by order of 2 (512,256)
N_PROCS = 0
VALID_BATCH_SIZE = 256
PLASMID_PATH = "data/plasmid.json"
SEQ_SIZE = 24
NUM_EPOCHS = 1 #replace with 80
CUDA_DEVICE_ID = 0
lr = 0.005 # 0.001 for attention layers in coreBlock


dataprocessor = BHIDataProcessor(
    path_to_training_data=TRAIN_DATA_PATH,
    path_to_validation_data=VALID_DATA_PATH,
    train_batch_size=TRAIN_BATCH_SIZE,
    train_workers=N_PROCS,
    valid_batch_size=VALID_BATCH_SIZE,
    valid_workers=N_PROCS,
    shuffle_train=True,
    shuffle_val=False,
    plasmid_path=PLASMID_PATH,
    seqsize=SEQ_SIZE,
    generator=generator
)
# print(next(dataprocessor.prepare_train_dataloader()))
first = BHIFirstLayersBlock(in_channels=4, out_channels=320, seqsize=24, kernel_sizes=[9, 15], pool_size=1, dropout=0.2)
core = BHICoreBlock(in_channels=first.out_channels, out_channels=64, seqsize=first.infer_outseqsize())
final = BHIFinalLayersBlock(in_channels=core.out_channels, seqsize=core.infer_outseqsize())
model = PrixFixeNet(first=first, core=core, final=final,generator=generator)
summary(model, (1, 4, 24))

trainer = BHITrainer(
    model,
    device=torch.device(f"cpu:0"),
    model_dir="/Users/john/data/model_weights",
    dataprocessor=dataprocessor,
    num_epochs=NUM_EPOCHS)

trainer.fit()

from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from bgr.bhi.dataset import BHISeqDatasetProb
from torch.utils.data import DataLoader
test_df = pd.read_csv(f'{fdir}/{name}/{name}_test.csv')[["seq", "score"]]
train_df = pd.read_csv(f'{fdir}/{name}/{name}_train.csv')[["seq", "score"]]
predictor = BHIPredictor(model=model, model_pth='/Users/john/data/model_weights/model_best.pth',
                         train_df=train_df,device=torch.device(f"cpu"))
valid_ds = BHISeqDatasetProb(test_df,seqsize=24)
valid_dl = DataLoader(valid_ds,batch_size=256,num_workers=0, shuffle=False)
pred_expr = []
for bs in tqdm(valid_dl):
    pred_expr.append(predictor.predict(bs))
pred_expr = np.concatenate(pred_expr, axis=0)

print(pearsonr(pred_expr, list(test_df.iloc[:, 1])), spearmanr(pred_expr, list(test_df.iloc[:, 1])))
