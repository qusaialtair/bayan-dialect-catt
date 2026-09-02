# Bayan Dialect-CATT — Arabic Dialect Diacritization

Fine-tuned [CATT](https://github.com/abjadai/catt) models for **Gulf, Egyptian, and Levantine** Arabic diacritization (tashkeel / تشكيل). The first practical per-dialect diacritizers we know of — existing tools are MSA-only, and CAMeL-Tools' dialect disambiguators frequently return input text *undiacritized* for Gulf and Levantine, which breaks downstream TTS and text-to-phoneme pipelines.

Built as the text frontend of [Bayan](https://github.com/) — an open multi-dialect Arabic TTS project.

## Results

DER (diacritic error rate, no case endings) on held-out, hand-written dialect gold:

| dialect | CATT (MSA, forced) | CAMeL-Tools dialect model | **Bayan dialect-CATT** |
|---|---|---|---|
| Gulf (Saudi) | 11.7% | frequently returns undiacritized text | **2.5%** |
| Egyptian | 14.7% | decent but 12× slower | **1.6%** |
| Levantine | 17.0% | frequently returns undiacritized text | **1.5%** |

For reference, CAMeL on MSA measures ~22% DER on the same metric family; the original CATT-EO reaches ~2.5% on MSA.

## Method

1. **Hand-written gold seeds** (~220–230 sentences per dialect, incl. a held-out test split) with careful dialect phonology — no case endings, no tanwin, dialect-specific reflexes (ودِّي/الحِين Gulf, بِدِّي/هَلَّق Levantine, دَلْوَقْتِي/عَايِز Egyptian).
2. **Silver labels** (4–11k pool sentences per dialect): base CATT predictions passed through *de-MSA-ization* rules — tanwin stripping, case-ending removal, de-hamzaization after وَ/فَ prefixes, and per-dialect frequent-word overrides — then an integrity filter (letters must be unchanged, sane diacritic ratio).
3. **Fine-tune** CATT-EO (encoder-only, 6 layers, 75MB) on silver + 8×-weighted gold, 4–6 epochs on a single consumer GPU (~10 min per dialect).

Everything is reproducible from this repo: seeds in `seeds/`, the full pipeline in `src/`.

## Use

```python
# after cloning https://github.com/abjadai/catt and downloading our checkpoints (see Releases):
import torch, sys
sys.path.insert(0, "catt")
from tashkeel_tokenizer import TashkeelTokenizer
from eo_pl import TashkeelModel
from utils import remove_non_arabic

tok = TashkeelTokenizer()
model = TashkeelModel(tok, max_seq_len=1024, n_layers=6, learnable_pos_emb=False)
model.load_state_dict(torch.load("catt_gulf.pt", map_location="cpu", weights_only=True))
model.eval()

text = "شلونك يا بو خالد عساك طيب"   # undiacritized Gulf
out = model.do_tashkeel_batch([remove_non_arabic(text)], 8, False)
print(out[0])                        # شْلُونِكْ يَا بُو خَالِدْ عَسَاك طَيِّبْ
```

Checkpoints (75MB each): see [Releases](../../releases) — `catt_gulf.pt`, `catt_egyptian.pt`, `catt_levantine.pt`.

## Reproduce / extend to a new dialect

```bash
git clone https://github.com/abjadai/catt
# write seeds/<dialect>_seed.json (train + test), extend de_msa_rules.py overrides
python src/gen_silver.py            # silver from any dialect text corpus
python src/fine_tune_catt.py gulf 6
python src/eval_catt_dialect.py gulf
```

## Credits & license

- Base model: [CATT: Character-based Arabic Tashkeel Transformer](https://arxiv.org/abs/2407.03236) by AbjadAI (Apache-2.0).
- Gold seeds written for this project; silver-labeling pipeline included.
- Released under **Apache-2.0**. If the models help your work, a star and a mention of Bayan are appreciated.
