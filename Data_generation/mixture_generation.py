import csv
import random
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# ---------------- CONFIG ---------------- #

ISOLATED_CSV = Path("/home/omkar/datasets/FSD50K/isolated_events_labels.csv")
ISOLATED_AUDIO_DIR = Path("/home/omkar/datasets/FSD50K/isolated_events_audio") 
BACKGROUND_DIR = Path("/home/omkar/datasets/TUTsoundevents2017devset/street")

OUTPUT_DIR = Path("/home/omkar/datasets/mixture_data")
OUTPUT_AUDIO_DIR = OUTPUT_DIR / "audio"
OUTPUT_CSV_PATH = OUTPUT_DIR / "mixture_labels.csv"

TARGET_SR = 44100           # Target sampling rate
CLIP_LEN_SEC = 10.0         # Length of final mixture clips in seconds
BACKGROUND_RMS_DBFS = -26.0 # Target RMS level for background audio in dBFS
SEED = 42                   # Random seed for reproducibility

OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

# ---------------- Functions ---------------- #

# Convert to mono
def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio[:, 0] # take first channel

def normalize_peak_0dbfs(audio: np.ndarray) -> np.ndarray: # calculate gain and set peak to 0dbfs
    """
    Scales audio so that its maximum absolute amplitude is 1.0 (0 dBFS).
    """
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio / peak

def peak_amplitude(audio: np.ndarray) -> float: # get peak amplitude temp jst for checking max amps before mixing
    return np.max(np.abs(audio))

def set_rms_dbfs(audio: np.ndarray, target_dbfs: float) -> np.ndarray: # calculate gain and set rms
    """
    Scales audio so its RMS energy matches the target_dbfs.
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms == 0:
        return audio 

    # Calculate gain needed: Target_Linear / Current_Linear
    # We convert dB back to linear amplitude: 10^(dB/20)
    target_linear = 10 ** (target_dbfs / 20)
    gain = target_linear / (current_rms + 1e-9)
    return audio * gain

def random_slice_background(audio: np.ndarray, sr: int, length_sec: float) -> np.ndarray:
    length_samples = int(length_sec * sr)
    if len(audio) <= length_samples:
        padding = length_samples - len(audio)
        return np.pad(audio, (0, padding))
    
    max_start = len(audio) - length_samples
    start = random.randint(0, max_start)
    return audio[start:start + length_samples]

def to_16bit_pcm(audio: np.ndarray) -> np.ndarray:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)

# ---------------- Main pipeline ---------------- #

# 1. LOAD BACKGROUNDS
print("--- Step 1: Loading Backgrounds ---")
background_files = list(BACKGROUND_DIR.glob("*.wav"))
loaded_backgrounds = []

for bg_path in tqdm(background_files, desc="Loading BG Assets"):
    audio, sr = sf.read(bg_path)
    audio = to_mono(audio)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    loaded_backgrounds.append(audio)

if not loaded_backgrounds:
    raise ValueError("No background files found!")

# 2. READ ISOLATED EVENTS
print("\n--- Step 2: Reading Isolated Events List ---")
event_records = []
with open(ISOLATED_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        event_records.append(row)

# 3. GENERATE MIXTURES
print("\n--- Step 3: Mixing & Generating Dataset ---")

mixture_metadata = [] # to hold metadata for CSV
cropped_events = [] # to log strictly cropped events

buffer_samples = int(0.1 * TARGET_SR) # 100 ms buffer on each side
clip_len_samples = int(CLIP_LEN_SEC * TARGET_SR) # total samples in clip
max_allowed_event_len = clip_len_samples - (2 * buffer_samples) # max event length to fit with buffer

for i, event_row in enumerate(tqdm(event_records, desc="Mixing")):
    
    # --- PREPARE FILENAME ---
    filename = event_row["fname"] + ".wav"
    label = event_row["label"]
    event_path = ISOLATED_AUDIO_DIR / filename
    
    try: # read event audio
        ev_audio, sr = sf.read(event_path) # read file
    except Exception as e: 
        print(f"\n[Error] Could not read {filename}: {e}") 
        continue

    ev_audio = to_mono(ev_audio) # convert to mono

    if sr != TARGET_SR: # resample if needed
        ev_audio = librosa.resample(ev_audio, orig_sr=sr, target_sr=TARGET_SR) #sample rate to target sample rate only a safeguard
    
    # normalise isolated events to 0dbfs 
    ev_audio = normalize_peak_0dbfs(ev_audio)
    
    event_len_samples = len(ev_audio) # length of event in samples

    # --- STRICT CROP LOGIC --- ***********need to be removed later
    if event_len_samples > max_allowed_event_len:
        orig_duration = event_len_samples / TARGET_SR
        max_allowed_duration = max_allowed_event_len / TARGET_SR
        
        ev_audio = ev_audio[:max_allowed_event_len]
        event_len_samples = len(ev_audio)
        
        print(f"\n>> WARNING: Cropped '{filename}' strictly. (Was {orig_duration:.2f}s, Now {max_allowed_duration:.2f}s)")
        cropped_events.append({"filename": filename, "orig": orig_duration})

    # --- PREPARE BACKGROUND ---
    bg_source = random.choice(loaded_backgrounds)
    bg_slice = random_slice_background(bg_source, TARGET_SR, CLIP_LEN_SEC)
    
    # Force Background to exact RMS (-26 dB)
    bg_slice = set_rms_dbfs(bg_slice, BACKGROUND_RMS_DBFS)

    # --- MIXING LOGIC ---
    min_start = buffer_samples # 100 ms buffer
    max_start = (clip_len_samples - event_len_samples) - buffer_samples # ensure event fits with buffer
    if max_start < min_start: max_start = min_start # edge case safeguard

    start_index = random.randint(min_start, max_start)
    end_index = start_index + event_len_samples

    # Superimpose
    mixture = bg_slice.copy()
    mixture[start_index:end_index] += ev_audio

    # --- CHECK CLIPPING ---
    # This is CRITICAL now. Since we are scaling by RMS, the peaks might be > 1.0.
    # The clipping guard ensures we preserve the SNR ratio even if we have to volume down.
    mix_peak = np.max(np.abs(mixture))
    if mix_peak > 1.0:
        mixture = mixture / mix_peak

    # --- SAVE ---
    mix_filename = f"mix_{i:06d}.wav"
    out_path = OUTPUT_AUDIO_DIR / mix_filename
    mix_16bit = to_16bit_pcm(mixture)
    sf.write(out_path, mix_16bit, TARGET_SR, subtype='PCM_16')

    # RECORD METADATA
    onset_sec = start_index / TARGET_SR
    offset_sec = end_index / TARGET_SR

    mixture_metadata.append({
        "filename": mix_filename,
        "onset": round(onset_sec, 4),
        "offset": round(offset_sec, 4),
        "label": label
    })

# 4. SAVE CSV & REPORT
print("\n--- Step 4: Saving Metadata CSV ---")
fieldnames = ["filename", "onset", "offset", "label"]

with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(mixture_metadata)

print(f"Success! Generated {len(mixture_metadata)} files.")

if cropped_events:
    print(f"\n[!] FINAL REPORT: {len(cropped_events)} events were strictly cropped.")
    for item in cropped_events:
        print(f" - {item['filename']} (Original: {item['orig']:.2f}s)")
else:
    print("\n[OK] Perfect run. No events needed cropping.")