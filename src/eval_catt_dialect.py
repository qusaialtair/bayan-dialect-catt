#!/usr/bin/env python3
"""Eval dialect-CATT vs MSA-CATT baseline on the hand-written seed test."""
import sys

sys.path.insert(0, "/home/work/catt")
import torch
from tashkeel_tokenizer import TashkeelTokenizer
from eo_pl import TashkeelModel
from utils import remove_non_arabic

tok = TashkeelTokenizer()
DIALECT = sys.argv[1]
CKPT = f"/home/work/catt/models/catt_{DIALECT}.pt"
TEST = f"/home/work/catt/dialect_data/{DIALECT}/seed_test.txt"

gold = [l.strip() for l in open(TEST, encoding="utf-8") if l.strip()]
stripped = [tok.remove_tashkeel(tok.clean_text(g)) for g in gold]


def load(path):
    m = TashkeelModel(tok, max_seq_len=1024, n_layers=6, learnable_pos_emb=False)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


def der(model, name):
    outs = model.do_tashkeel_batch([remove_non_arabic(s) for s in stripped], 16, False)
    dist = ref = 0.0
    for g, h in zip(gold, outs):
        if not h:
            continue
        d = tok.compute_der(tok.clean_text(g), tok.clean_text(h), case_ending=False)
        dist += d["distance"]; ref += d["ref_length"]
    print(f"{DIALECT} {name:12s} DER(no-case) = {100*dist/max(ref,1):5.2f}%")
    return outs


outs_base = der(load("/home/work/catt/models/best_eo_mlm_ns_epoch_193.pt"), "MSA-CATT")
outs_ft = der(load(CKPT), "dialect-CATT")
for i in range(min(3, len(gold))):
    print(f"  s{i} gold : {gold[i][:56]}")
    print(f"     msa   : {outs_base[i][:56]}")
    print(f"     dial  : {outs_ft[i][:56]}")
