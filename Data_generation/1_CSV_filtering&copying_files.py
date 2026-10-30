import csv
import shutil

from tqdm import tqdm
from pathlib import Path

#this script filters csv to get (desired classes) with (duration <9.5 sec) and copies corresponding wav files to output folder

#edge trimming or energy envelope based trimming can be done later if needed. (bell and rattle have some leading/trailing silences)

# -------- CONFIG --------for csv genration with columns as filename,label,duration
INPUT_CSV = Path("/home/omkar/datasets/FSD50K/dev_with_duration.csv")
OUTPUT_CSV = Path("/home/omkar/datasets/FSD50K/isolated_events_labels.csv")

TARGET_CLASSES = {
    "Hiss",
    "Rattle",
    "Bell"
}

with open(INPUT_CSV, newline="", encoding="utf-8") as infile, \
     open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)

    # write header
    writer.writerow(["fname", "label", "duration"])

    for row in reader:
        # duration handling 
        dur_str = row.get("duration")

        # skip rows with missing duration
        if not dur_str:
            continue

        try:
            dur = float(dur_str) #string to float
        except ValueError:
            continue

        # keep only duration < 9.5
        if dur >= 9.5:
            continue

        labels = row["labels"].split(",")

        ################### keep only single class rows for later needs to be handled because some classes have 2 labels but one event like "Finger_snapping,Snapping". ###########
        # ***********Need to actually check audio to decide.********* 
        if len(labels) != 1:
            continue

        label = labels[0].strip()

        if label not in TARGET_CLASSES:
            continue

        writer.writerow([row["fname"], label, round(dur, 3)])

print("Done. Filtered CSV written to:", OUTPUT_CSV)


# -------- CONFIG --------coying files as per above generated csv
ROOT_AUDIO_DIR = Path("/home/omkar/datasets/FSD50K/dev/FSD50K.dev_audio")   # where all wavs exist
CSV_FILE = Path("/home/omkar/datasets/FSD50K/isolated_events_labels.csv")  # filtered csv with filenames to copy
OUTPUT_DIR = Path("/home/omkar/datasets/FSD50K/isolated_events_audio")  # where to copy the wavs
AUDIO_EXT = ".wav"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

copied = 0
missing = 0
skipped = 0

# First: count total rows for progress bar
with open(CSV_FILE, newline="", encoding="utf-8") as f:
    total_rows = sum(1 for _ in f) - 1  # minus header

with open(CSV_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in tqdm(reader, total=total_rows, desc="Copying WAV files"):
        fname = row["fname"] + AUDIO_EXT

        # search recursively in root folder
        matches = list(ROOT_AUDIO_DIR.rglob(fname))

        if not matches:
            missing += 1
            continue

        src = matches[0]  # take first match
        dst = OUTPUT_DIR / fname

        if dst.exists(): # skip if already exists
            skipped += 1
            continue

        shutil.copy2(src, dst)  # COPY, not move
        copied += 1

print(f"Copied files: {copied}")
print(f"Missing files: {missing}")
print(f"Skipped files (already exist): {skipped}")
print("Done.")