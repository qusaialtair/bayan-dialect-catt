#!/usr/bin/env python3
"""De-MSA-ization rules: turn CATT (MSA) diacritization into dialect-flavored
silver labels. Used only for GENERATING fine-tuning data, never at serving.

Rules:
  1. strip tanwin (dialects have none)
  2. strip formal case endings (reuse the production spoken-waqf module)
  3. drop hamza after وَ/فَ prefixes (وأشتري -> وَشْتَري)
  4. per-dialect frequent-word overrides (hand-written, from pool frequency)
Integrity is enforced by the caller (letters must be unchanged).
"""
import re
import sys

sys.path.insert(0, "/mnt/c/Users/Work/Documents/Bayan-Project/bayan_engine")
from text_frontend.spoken_normalizer import SpokenNormalizer

_waqf = SpokenNormalizer()

TANWIN = "\u064b\u064c\u064d"

OVERRIDES = {
    "gulf": {
        "شلون": "شْلُون", "شلونك": "شْلُونِك", "وشلون": "وِشْلُون",
        "الحين": "الحِين", "وايد": "وَايِد", "زين": "زَيْن", "كذا": "كِذَا",
        "موف": "مُو", "مو": "مُو", "بعدين": "بَعْدِين", "يبي": "يِبِي",
        "ابغي": "أَبْغَى", "ابغى": "أَبْغَى", "ودي": "وِدِّي", "تبي": "تِبِي",
        "مافي": "مَا فِي", "عساكم": "عَسَاكُم", "هلا": "هَلَا", "فديتك": "فِدِيتِك",
        "قلت": "قُلْت", "شوي": "شْوَي", "يستاهل": "يِسْتَاهِل", "خلك": "خَلِّك",
    },
    "egyptian": {
        "دلوقتي": "دَلْوَقْتِي", "عايز": "عَايِز", "عايزة": "عَايْزَة",
        "مش": "مُش", "ازيك": "إِزَّيْك", "ازيكم": "إِزَّيْكُم", "ايه": "إِيه",
        "كده": "كِدَه", "النهارده": "النَّهَارْدَه", "بكرة": "بُكْرَة",
        "يلا": "يَالَا", "هنروح": "هِنِرُوح", "مفيش": "مَفِيش",
        "عشان": "عَشَان", "كده": "كِدَه", "دول": "دُول", "بيبقى": "بِيِبْقَى",
        "حصل": "حَصَل", "خالص": "خَالِص", "اوي": "أَوِي", "يعني": "يَعْنِي",
    },
    "levantine": {
        "بدنا": "بَدَّنا", "هيك": "هَيْك", "هلق": "هَلَّق", "شو": "شُو",
        "كتير": "كْتِير", "منيح": "مْنِيح", "كيفك": "كِيفَك", "ينع": "يْنِع",
        "هلقلي": "هَلَّقْلِي", "مش": "مِش", "في": "فِي", "مافي": "مَا فِي",
        "عم": "عَم", "بدي": "بِدِّي", "بدك": "بِدَّك", "عال": "عَال",
        "زبط": "زْبَط", "حكي": "حِكِي", "دغري": "دْغَرِي", "تطول": "تْطُول",
    },
}


def de_msa(text: str, dialect: str) -> str:
    out = "".join(c for c in text if c not in TANWIN)
    out = re.sub(r"\u064e\u0623", "\u064e", out)   # وَأَ / فَأَ -> وَ / فَ
    out = _waqf.apply_spoken_waqf(out)
    ov = OVERRIDES.get(dialect, {})
    words = out.split(" ")
    fixed = []
    for w in words:
        bare = re.sub(r"[\u064b-\u0652]", "", w).strip("،.!?؟")
        if bare in ov:
            w = ov[bare]
        fixed.append(w)
    return " ".join(fixed)
