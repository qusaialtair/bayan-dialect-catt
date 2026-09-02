#!/usr/bin/env python3
"""Generate dialect silver labels from the pool: CATT(MSA) -> de-MSA rules ->
integrity filter. Output: CATT dataset folders + seed eval files."""
import json
import os
import random
import re
import sys

sys.path.insert(0, "/home/work/catt")
sys.path.insert(0, "/mnt/c/Users/Work/Documents/Bayan-Project/scripts")

from de_msa_rules import de_msa
from tashkeel_tokenizer import TashkeelTokenizer
import torch
from eo_pl import TashkeelModel
from utils import remove_non_arabic

MANIFEST = "/mnt/c/Users/Work/kokoro_arabic/data_pipeline/dataset/master_manifest.jsonl"
SEEDS = "/mnt/c/Users/Work/Documents/Bayan-Project/bayan_engine/catt_seeds"
OUT = "/home/work/catt/dialect_data"
N_SILVER = 12000
ONLY = os.environ.get("CATT_ONLY_DIALECT")  # optional: one dialect
SEED_WEIGHT = 8          # repeat hand-written gold this many times
random.seed(61)

tok = TashkeelTokenizer()


def bare(t):
    t = tok.clean_text(t)
    t = tok.remove_tashkeel(t)
    return re.sub(r"[آأإ]", "ا", t).strip()


def ratio_ok(t):
    a = sum(1 for c in t if "\u064b" <= c <= "\u0652")
    n = sum(1 for c in t if "\u0621" <= c <= "\u064a")
    return n > 0 and 0.2 <= a / n <= 0.95


def main():
    # pool sentences per dialect (undiacritized, deduped)
    seen = set()
    pools = {d: [] for d in ("gulf", "egyptian", "levantine")}
    for line in open(MANIFEST, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        d = r.get("dialect")
        if d not in pools:
            continue
        t = (r.get("text") or "").strip()
        if not (12 <= len(t) <= 130):
            continue
        nonspace = [c for c in t if not c.isspace()]
        if not nonspace or sum("\u0600" <= c <= "\u06FF" for c in nonspace) < len(nonspace) * 0.85:
            continue
        t = re.sub(r"\s+", " ", t).strip(" ،,.!?؟؛:)")
        k = bare(t)
        if not k or k in seen:
            continue
        seen.add(k)
        pools[d].append(t)
        if all(len(v) >= N_SILVER for v in pools.values()):
            break
    for d, v in pools.items():
        print(f"pool {d}: {len(v)} unique sentences", flush=True)

    model = TashkeelModel(tok, max_seq_len=1024, n_layers=6,
                          learnable_pos_emb=False)
    model.load_state_dict(torch.load(
        "/home/work/catt/models/best_eo_mlm_ns_epoch_193.pt",
        map_location="cpu", weights_only=True))
    model.eval()

    for d, sents in pools.items():
        if ONLY and d != ONLY:
            continue
        random.shuffle(sents)
        sents = sents[:N_SILVER]
        preds = model.do_tashkeel_batch([remove_non_arabic(s) for s in sents],
                                        64, False)
        kept = []
        for src, hyp in zip(sents, preds):
            lab = de_msa(hyp, d)
            if bare(lab) != bare(src):      # letters must be unchanged
                continue
            if not ratio_ok(lab):
                continue
            kept.append(lab)
        print(f"{d}: silver kept {len(kept)}/{len(sents)}", flush=True)

        seed = json.load(open(os.path.join(SEEDS, f"{d}_seed.json"), encoding="utf-8"))
        seed2_path = os.path.join(SEEDS, f"{d}_seed2.json")
        if os.path.exists(seed2_path):
            seed2 = json.load(open(seed2_path, encoding="utf-8"))
            seed["train"] = seed["train"] + seed2["train"]
            seed["test"] = seed["test"] + seed2["test"]
        train_dir = os.path.join(OUT, d, "train")
        val_dir = os.path.join(OUT, d, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        random.shuffle(kept)
        n_val = min(300, len(kept) // 20)
        with open(os.path.join(train_dir, "silver.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(kept[n_val:]) + "\n")
        with open(os.path.join(train_dir, "seeds.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(seed["train"] * SEED_WEIGHT) + "\n")
        with open(os.path.join(val_dir, "val.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(kept[:n_val] + seed["train"][:15]) + "\n")
        with open(os.path.join(OUT, d, "seed_test.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(seed["test"]) + "\n")
        print(f"{d}: wrote train/val ({len(kept)-n_val} silver + {len(seed['train'])*SEED_WEIGHT} seed rows)", flush=True)


if __name__ == "__main__":
    main()
