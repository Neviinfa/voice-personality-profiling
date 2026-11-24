
import os
import numpy as np
import soundfile as sf

folder = "dataset/augmented_train"

files = [
    f for f in os.listdir(folder)
    if f.lower().endswith(".wav")
]

invalid = []
empty = []

for filename in files:
    path = os.path.join(folder, filename)

    try:
        audio, sr = sf.read(path)

        if len(audio) == 0:
            empty.append(filename)
        elif not np.isfinite(audio).all():
            invalid.append(filename)

    except Exception:
        invalid.append(filename)

print("Total WAV files:", len(files))
print("Empty files:", len(empty))
print("Invalid files:", len(invalid))
print("Verification:", "PASSED" if not empty and not invalid else "CHECK REQUIRED")
