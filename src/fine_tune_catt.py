#!/usr/bin/env python3
"""Fine-tune CATT-EO on a dialect's silver+gold data. Usage: fine_tune_catt.py <dialect> [epochs]"""
import os
import sys

sys.path.insert(0, "/home/work/catt")
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tashkeel_tokenizer import TashkeelTokenizer
from eo_pl import TashkeelModel
from tashkeel_dataset import TashkeelDataset, PrePaddingDataLoader

DIALECT = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
DATA = f"/home/work/catt/dialect_data/{DIALECT}"
BASE = "/home/work/catt/models/best_eo_mlm_ns_epoch_193.pt"

tok = TashkeelTokenizer()
train_ds = TashkeelDataset(f"{DATA}/train", tok, max_seq_len=256,
                           tashkeel_to_text_ratio_threshold=0.12)
val_ds = TashkeelDataset(f"{DATA}/val", tok, max_seq_len=256,
                         tashkeel_to_text_ratio_threshold=0.12)
print(f"train={len(train_ds)} val={len(val_ds)}", flush=True)
train_dl = PrePaddingDataLoader(tok, train_ds, batch_size=32, shuffle=True, num_workers=2)
val_dl = PrePaddingDataLoader(tok, val_ds, batch_size=32, num_workers=2)

model = TashkeelModel(tok, max_seq_len=256, n_layers=6, learnable_pos_emb=False)
model.load_state_dict(torch.load(BASE, map_location="cpu", weights_only=True))

trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="gpu", devices=1,
                     precision=32, gradient_clip_val=1.0, log_every_n_steps=20,
                     default_root_dir=f"/home/work/catt/ft_{DIALECT}")
trainer.fit(model, train_dl, val_dl)

out = f"/home/work/catt/models/catt_{DIALECT}.pt"
torch.save(model.state_dict(), out)
print(f"saved {out}", flush=True)
