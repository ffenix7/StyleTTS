import os
from pathlib import Path
import random

ESD_ROOT = Path("./ESD")
SPEAKER = "0011"
TRANSCRIPT_PATH = ESD_ROOT / SPEAKER / (SPEAKER + ".txt")
OUT_TRAIN = "Data/ESD/train_list_esd_0011.txt"
OUT_VAL = "Data/ESD/val_list_esd_0011.txt"

wav_paths = sorted((ESD_ROOT / SPEAKER).glob("**/*.wav"))
print(f"Found {len(wav_paths)} wav files")


def load_metadata():
    meta = {}
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            utt_id, transcript, emotion = parts
            meta[utt_id] = (transcript, emotion)
    print(f"Loaded metadata for {len(meta)} recordings")
    return meta


metadata = load_metadata()

entries = []
for wav in wav_paths:
    utt_id = wav.stem
    if utt_id not in metadata:
        print(f"Warning: no metadata for {utt_id}")
        continue

    transcript, emotion = metadata[utt_id]
    transcript = transcript.replace("|", " ")
    entries.append(f"{wav}|{transcript}|0\n")

random.shuffle(entries)
split = int(0.9 * len(entries))
train_entries = entries[:split]
val_entries = entries[split:]

os.makedirs(Path(OUT_TRAIN).parent, exist_ok=True)
with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    f.writelines(train_entries)
with open(OUT_VAL, "w", encoding="utf-8") as f:
    f.writelines(val_entries)

print(f"Saved {len(train_entries)} train, {len(val_entries)} val")