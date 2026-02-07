import shutil
import random
import math
import pandas as pd
import torchaudio

from pathlib import Path
from tqdm import tqdm

"""
Generate FDY-SED compatible dataset splits with:
- original (44.1 kHz) WAVs
- resampled 16 kHz WAVs
- strong / weak labels
- duration files
"""

# ================= CONFIG =================
RANDOM_SEED = 42

INPUT_WAV_DIR = Path("/home/omkar/datasets/mixture_data/audio")
INPUT_LABELS = Path("/home/omkar/datasets/mixture_data/mixture_labels.csv")

OUTPUT_ROOT = Path("/home/omkar/datasets/dataset_splits")

TARGET_SR = 16000

PCT_STRONG = 0.70
PCT_WEAK = 0.15
PCT_UNLABELED = 0.05
PCT_EVAL = 0.10
# ==========================================

assert abs(PCT_STRONG + PCT_WEAK + PCT_UNLABELED + PCT_EVAL - 1.0) < 1e-6
random.seed(RANDOM_SEED)

# ================= LOAD & NORMALIZE LABELS =================
df = pd.read_csv(INPUT_LABELS)

# 🔑 FDY-SED expects `event_label`
if "label" in df.columns:
    df = df.rename(columns={"label": "event_label"})

required_cols = {"filename", "onset", "offset", "event_label"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in labels CSV: {missing}")

files = sorted(df["filename"].unique())
random.shuffle(files)
N = len(files)

# ================= SPLIT FILES =================
n_strong = math.floor(PCT_STRONG * N)
n_weak = math.floor(PCT_WEAK * N)
n_unlabeled = math.floor(PCT_UNLABELED * N)

strong_files = files[:n_strong]
weak_files = files[n_strong : n_strong + n_weak]
unlabeled_files = files[n_strong + n_weak : n_strong + n_weak + n_unlabeled]
eval_files = files[n_strong + n_weak + n_unlabeled :]

splits = {
    "synth_train": strong_files,
    "synth_val": eval_files[: len(eval_files) // 2],
    "test": eval_files[len(eval_files) // 2 :],
    "weak_train": weak_files,
    "unlabeled_train": unlabeled_files,
}

# ================= CREATE DIRECTORIES =================
for split in splits:
    (OUTPUT_ROOT / split / "wav_orig").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / split / "wav_16k").mkdir(parents=True, exist_ok=True)

# ================= AUDIO PROCESSING =================
resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=TARGET_SR)
durations = []

for split, flist in splits.items():
    print(f"\nProcessing {split}...")
    for fname in tqdm(flist, unit="file"):
        src = INPUT_WAV_DIR / fname
        if not src.exists():
            raise FileNotFoundError(f"Missing WAV file: {src}")

        # copy original WAV
        dst_orig = OUTPUT_ROOT / split / "wav_orig" / fname
        shutil.copy(src, dst_orig)

        # load + resample
        wav, sr = torchaudio.load(src)
        if sr != TARGET_SR:
            wav = resampler(wav)

        dst_16k = OUTPUT_ROOT / split / "wav_16k" / fname
        torchaudio.save(dst_16k, wav, TARGET_SR)

        durations.append({
            "filename": fname,
            "duration": wav.shape[1] / TARGET_SR
        })

dur_df = pd.DataFrame(durations)

# ================= STRONG LABELS + DURATIONS =================
for split in ["synth_train", "synth_val", "test"]:
    strong_tsv = OUTPUT_ROOT / split / "strong_labels.tsv"
    duration_tsv = OUTPUT_ROOT / split / "durations.tsv"

    df[df["filename"].isin(splits[split])][
        ["filename", "onset", "offset", "event_label"]
    ].to_csv(strong_tsv, sep="\t", index=False, header=False)

    dur_df[dur_df["filename"].isin(splits[split])].to_csv(
        duration_tsv, sep="\t", index=False, header=False
    )

# ================= WEAK LABELS =================
weak_df = (
    df[df["filename"].isin(weak_files)]
    .groupby("filename")["event_label"]
    .apply(lambda x: ",".join(sorted(set(x))))
    .reset_index()
)

# write as: filename <tab> event_labels
weak_df.to_csv(
    OUTPUT_ROOT / "weak_train" / "weak_labels.tsv",
    sep="\t",
    index=False,
    header=False
)

# ================= SUMMARY =================
print("\n================ SUMMARY ================")
print(f"Total files: {N}")
for split, flist in splits.items():
    print(f"{split:16s}: {len(flist):6d} ({len(flist)/N:.2%})")
print(f"\nDataset written to: {OUTPUT_ROOT}")
