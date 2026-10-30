import csv
import random
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# ---------------- CONFIG ---------------- #

ISOLATED_CSV        = Path("/home/dll1303/Omkar/Dataset_root/isolatedEvents_5classes/isolated_events_5classes_labels.csv")
ISOLATED_AUDIO_DIR  = Path("/home/dll1303/Omkar/Dataset_root/isolatedEvents_5classes/isolated_events_audio_5classes")
BACKGROUND_DIR      = Path("/home/dll1303/Omkar/Dataset_root/background_noise/street")

OUTPUT_DIR          = Path("/home/dll1303/Omkar/Dataset_root/mixture_data_multiple_non_overlapping")
OUTPUT_AUDIO_DIR    = OUTPUT_DIR / "audio"
OUTPUT_CSV_PATH     = OUTPUT_DIR / "mixture_labels.csv"

TARGET_SR           = 16000     # Target sampling rate (Hz)
CLIP_LEN_SEC        = 10.0      # Duration of each output mixture clip (seconds)
BACKGROUND_RMS_DBFS = -20.0     # Target RMS level for background noise (dBFS)
NUM_MIXTURES        = 8000      # Total number of mixture clips to generate
MAX_EVENTS_PER_CLIP = 5         # Maximum isolated events placed in one clip (minimum is always 1)
MAX_FIT_ATTEMPTS    = 100       # Max random-draw attempts to fit an event into remaining space
BUFFER_SEC          = 0.01      # Hard silence buffer at clip start and end (seconds)
GAP_SEC             = 0.50      # Minimum gap between the buffer/events (seconds)
SEED                = 42        # Random seed for reproducibility

OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

# Derived constants (samples)
CLIP_LEN_SAMPLES = int(CLIP_LEN_SEC * TARGET_SR)
BUFFER_SAMPLES   = int(BUFFER_SEC   * TARGET_SR)
GAP_SAMPLES      = int(GAP_SEC      * TARGET_SR)

# Usable region: [BUFFER_SAMPLES  ...  CLIP_LEN_SAMPLES - BUFFER_SAMPLES)
USABLE_START = BUFFER_SAMPLES
USABLE_END   = CLIP_LEN_SAMPLES - BUFFER_SAMPLES


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def to_mono(audio: np.ndarray) -> np.ndarray:
    """Return first channel if multichannel, else pass through."""
    if audio.ndim == 1:
        return audio
    return audio[:, 0]


def normalize_peak_0dbfs(audio: np.ndarray) -> np.ndarray:
    """Scale audio so the peak absolute amplitude equals 1.0 (0 dBFS)."""
    peak = np.max(np.abs(audio))
    if peak == 0.0:
        return audio
    return audio / peak


def set_rms_dbfs(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale audio so its RMS matches target_dbfs."""
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms == 0.0:
        return audio
    target_linear = 10.0 ** (target_dbfs / 20.0)
    gain = target_linear / (current_rms + 1e-9)
    return audio * gain


def random_slice(audio: np.ndarray, length_samples: int) -> np.ndarray:
    """Cut a random window of `length_samples` from `audio`, padding if too short."""
    if len(audio) <= length_samples:
        return np.pad(audio, (0, length_samples - len(audio)))
    start = random.randint(0, len(audio) - length_samples)
    return audio[start : start + length_samples]


def to_16bit_pcm(audio: np.ndarray) -> np.ndarray:
    """Clip to [-1, 1] and convert to signed 16-bit PCM integer array."""
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


# ─────────────────────────────────────────────────────────────────────────────
# FIT-CHECKER
# ─────────────────────────────────────────────────────────────────────────────

def find_placement(event_len: int, occupied: list) -> int | None:
    """
    Place an event of `event_len` samples inside the usable region.

    Layout enforced:
        [BUFFER][GAP][event_1][GAP] ... [GAP][event_n][GAP][BUFFER]

    - First event  (occupied is empty)  → always placed at USABLE_START + GAP_SAMPLES
      i.e. the fixed position right after [buffer][gap].
    - Subsequent events → placed randomly in the first free slot that fits,
      starting immediately after [last_event_end + GAP].

    `occupied` is a sorted list of (start, end) tuples for already-placed events.

    Returns the chosen start sample index, or None if the event does not fit.
    """
    # ── First event: fixed position right after [buffer][gap] ───────────────
    if not occupied:
        fixed_start = USABLE_START + GAP_SAMPLES
        fixed_end   = fixed_start + event_len
        # Check it actually fits before the right buffer+gap wall
        if fixed_end <= USABLE_END - GAP_SAMPLES:
            return fixed_start
        return None   # event is too long even for an empty clip

    # ── Subsequent events: find the next available slot after all placed events
    # The candidate start is right after the last placed event + GAP
    last_end    = occupied[-1][1]                   # end of last placed event
    cand_start  = last_end + GAP_SAMPLES            # next position respecting gap
    cand_end    = cand_start + event_len

    # Check it fits before the right buffer+gap wall
    right_wall  = USABLE_END - GAP_SAMPLES
    if cand_end <= right_wall:
        return cand_start

    return None   # no room left in the clip


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – LOAD BACKGROUND FILES
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Step 1 : Loading background files")
print("=" * 60)

bg_files = list(BACKGROUND_DIR.glob("*.wav"))
if not bg_files:
    raise FileNotFoundError(f"No .wav files found in {BACKGROUND_DIR}")

loaded_backgrounds = []
for bg_path in tqdm(bg_files, desc="  BG files"):
    audio, sr = sf.read(bg_path)
    audio = to_mono(audio)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    loaded_backgrounds.append(audio)

print(f"  Loaded {len(loaded_backgrounds)} background file(s).\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – PRE-LOAD ALL ISOLATED EVENTS
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Step 2 : Loading isolated events")
print("=" * 60)

event_records = []
with open(ISOLATED_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        event_records.append(row)

# class_pool: {"label": [{"fname", "label", "audio"}, ...], ...}
loaded_events = {}
skipped = 0

for row in tqdm(event_records, desc="  Event files"):
    fname = row["fname"] + ".wav"
    label = row["label"]
    path  = ISOLATED_AUDIO_DIR / fname

    try:
        audio, sr = sf.read(path)
    except Exception as e:
        print(f"\n  [WARN] Cannot read {fname}: {e} – skipped.")
        skipped += 1
        continue

    audio = to_mono(audio)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # Scale isolated event to peak 0 dBFS
    audio = normalize_peak_0dbfs(audio)

    # Group by class label for balanced sampling
    if label not in loaded_events:
        loaded_events[label] = []
    loaded_events[label].append({"fname": fname, "label": label, "audio": audio})

total_loaded = sum(len(v) for v in loaded_events.values())
print(f"  Loaded {total_loaded} event(s) across {len(loaded_events)} class(es), skipped {skipped}.")
print(f"  Class distribution in pool:")
for cls, evs in sorted(loaded_events.items()):
    print(f"    {cls:20s} -> {len(evs)} files")
print()

if not loaded_events:
    raise RuntimeError("No isolated events could be loaded – aborting.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – GENERATE MIXTURE CLIPS
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Step 3 : Generating mixture clips")
print("=" * 60)

mixture_metadata = []   # rows for the output CSV  (one row per placed event)
mix_counter      = 0
stats_events     = []   # how many events were placed per clip (for the report)

for _ in tqdm(range(NUM_MIXTURES), desc="  Mixtures"):

    # ── 3a. Prepare background (mono, 10 s, RMS = BACKGROUND_RMS_DBFS) ──────
    bg_raw   = random.choice(loaded_backgrounds)
    bg_slice = random_slice(bg_raw, CLIP_LEN_SAMPLES)
    bg_slice = to_mono(bg_slice)
    bg_slice = set_rms_dbfs(bg_slice, BACKGROUND_RMS_DBFS)

    mixture  = bg_slice.copy()

    # ── 3b. Place events one by one ─────────────────────────────────────────
    # occupied: sorted list of (start_sample, end_sample) for events already added
    occupied      = []
    placed_events = []   # metadata for events placed in this clip

    # Keep placing events until the clip is full or MAX_EVENTS_PER_CLIP is reached.
    # Stops naturally when nothing fits in the remaining space.
    for _ in range(MAX_EVENTS_PER_CLIP):
        placed = False

        for attempt in range(MAX_FIT_ATTEMPTS):
            # Class-balanced sampling: pick a random class first, then a random file from it
            random_class = random.choice(list(loaded_events.keys()))
            ev           = random.choice(loaded_events[random_class])
            ev_len       = len(ev["audio"])

            start = find_placement(ev_len, occupied)

            if start is None:
                # This event doesn't fit anywhere – try a different random event
                continue

            # Place the event onto the mixture
            end = start + ev_len
            mixture[start:end] += ev["audio"]

            # Register as occupied (sorted order maintained by sort below)
            occupied.append((start, end))
            occupied.sort(key=lambda x: x[0])

            placed_events.append({
                "start": start,
                "end":   end,
                "label": ev["label"],
            })
            placed = True
            break   # move on to filling the next event slot

        if not placed:
            # Couldn't fit any event after MAX_FIT_ATTEMPTS – clip is full enough
            break

    # ── 3c. Clipping guard (preserves SNR while avoiding hard clipping) ──────
    mix_peak = np.max(np.abs(mixture))
    if mix_peak > 1.0:
        mixture = mixture / mix_peak

    # ── 3d. Save audio ───────────────────────────────────────────────────────
    mix_filename = f"mix_{mix_counter:06d}.wav"
    out_path     = OUTPUT_AUDIO_DIR / mix_filename
    sf.write(out_path, to_16bit_pcm(mixture), TARGET_SR, subtype="PCM_16")

    # ── 3e. Record strong labels (one CSV row per placed event) ──────────────
    for ev_info in placed_events:
        mixture_metadata.append({
            "filename": mix_filename,
            "onset":    round(ev_info["start"] / TARGET_SR, 4),
            "offset":   round(ev_info["end"]   / TARGET_SR, 4),
            "label":    ev_info["label"],
        })

    stats_events.append(len(placed_events))
    mix_counter += 1


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 4 : Saving metadata CSV")
print("=" * 60)

fieldnames = ["filename", "onset", "offset", "label"]
with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(mixture_metadata)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n  CSV saved       : {OUTPUT_CSV_PATH}")
print(f"  Clips generated : {mix_counter}")
print(f"  CSV rows        : {len(mixture_metadata)}")
if stats_events:
    print(f"  Events / clip   : min={min(stats_events)}  "
          f"max={max(stats_events)}  "
          f"avg={sum(stats_events)/len(stats_events):.2f}")
print("\n[Done]")