"""
q3a_analysis.py — generates every plot/experiment required for Q3A using
the REAL provided song database (EE200 Project Song Database/*.mp3).
Produces PNGs into ./figs/

If you don't have the dataset locally, set USE_SYNTHETIC=True to fall
back to a synthetic test signal instead.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

sys.path.insert(0, os.path.dirname(__file__))
from fingerprint import (compute_spectrogram, find_peaks_2d, generate_hashes,
                          fingerprint_signal, load_audio, FingerprintDB,
                          match_query, build_database)
from self_test import make_song, add_noise, pitch_shift_resample

USE_SYNTHETIC = False
SONG_FOLDER = "songs_raw/EE200 Project Song Database"
DB_PATH = "song_db.pkl"
ANALYSIS_SONG = "Yesterday"     # song used for DFT/spectrogram/constellation plots
QUERY_SONG = "Yesterday"        # song used as the query in robustness tests
CLIP_START_SEC = 30
CLIP_DUR_SEC = 10

os.makedirs("figs", exist_ok=True)
FS = 11025

if USE_SYNTHETIC:
    x = make_song(seed=2, duration=20, fs=FS)
else:
    x, FS = load_audio(os.path.join(SONG_FOLDER, f"{ANALYSIS_SONG}.mp3"))
    # use a representative 20s excerpt (full song works too, just slower)
    x = x[: 20 * FS]

# 1. Plain DFT of the whole song -- shows WHAT but not WHEN
X = np.fft.rfft(x)
freqs = np.fft.rfftfreq(len(x), d=1/FS)
plt.figure(figsize=(8, 3))
plt.plot(freqs, 20*np.log10(np.abs(X) + 1e-9))
plt.xlim(0, 4000)
plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude (dB)")
plt.title("Single DFT of entire clip — frequency content only, no timing info")
plt.tight_layout(); plt.savefig("figs/01_full_dft.png", dpi=120); plt.close()

# 2. Spectrogram with a "good" window
f, t, Sxx_db = compute_spectrogram(x, FS, nperseg=4096)
plt.figure(figsize=(8, 4))
plt.pcolormesh(t, f, Sxx_db, shading="auto", cmap="magma")
plt.ylim(0, 4000)
plt.colorbar(label="dB")
plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram (nperseg=4096) — frequency content over time")
plt.tight_layout(); plt.savefig("figs/02_spectrogram_main.png", dpi=120); plt.close()

# 3. Short vs long window comparison
for nperseg, label in [(256, "short_256"), (8192, "long_8192")]:
    f_, t_, S_ = compute_spectrogram(x, FS, nperseg=nperseg)
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t_, f_, S_, shading="auto", cmap="magma")
    plt.ylim(0, 4000)
    plt.colorbar(label="dB")
    plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz)")
    plt.title(f"Spectrogram, nperseg={nperseg}")
    plt.tight_layout(); plt.savefig(f"figs/03_spectrogram_{label}.png", dpi=120); plt.close()

# 4. Constellation map
peaks = find_peaks_2d(Sxx_db, amp_min_db=-40, neighborhood=(20, 20))
plt.figure(figsize=(8, 4))
pf = [f[p[0]] for p in peaks]; pt = [t[p[1]] for p in peaks]
plt.scatter(pt, pf, s=8, c="black")
plt.ylim(0, 4000)
plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz)")
plt.title(f"Constellation map ({len(peaks)} peaks)")
plt.tight_layout(); plt.savefig("figs/04_constellation.png", dpi=120); plt.close()

# 5. Noise robustness curve (using the REAL 50-song database)
if os.path.exists(DB_PATH):
    db = FingerprintDB.load(DB_PATH)
else:
    db = build_database(SONG_FOLDER, db_path=DB_PATH, nperseg=4096,
                         neighborhood=(20, 20), amp_min_db=-60)

query_x_full, _ = load_audio(os.path.join(SONG_FOLDER, f"{QUERY_SONG}.mp3"))
query_x = query_x_full[int(CLIP_START_SEC*FS): int((CLIP_START_SEC+CLIP_DUR_SEC)*FS)]

snrs = [40, 30, 20, 15, 10, 5, 0, -5, -10, -15]
correct = []
for snr in snrs:
    noisy = add_noise(query_x, snr)
    best, hist, qres = match_query(noisy, FS, db, nperseg=4096, neighborhood=(20, 20), amp_min_db=-60)
    correct.append(1 if best == QUERY_SONG else 0)

plt.figure(figsize=(7, 3))
plt.plot(snrs, correct, "o-")
plt.gca().invert_xaxis()
plt.yticks([0, 1], ["Wrong/No match", "Correct match"])
plt.xlabel("SNR (dB)")
plt.title(f"Recognition success vs. added noise — query: '{QUERY_SONG}'")
plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figs/05_noise_robustness.png", dpi=120); plt.close()

# 6. Pitch-shift robustness curve (real database)
shifts = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4]
correct_shift = []
for s in shifts:
    shifted = pitch_shift_resample(query_x, FS, s)
    best, hist, qres = match_query(shifted, FS, db, nperseg=4096, neighborhood=(20, 20), amp_min_db=-60)
    correct_shift.append(1 if best == QUERY_SONG else 0)

plt.figure(figsize=(7, 3))
plt.plot(shifts, correct_shift, "o-", color="darkorange")
plt.yticks([0, 1], ["Wrong/No match", "Correct match"])
plt.xlabel("Pitch shift (semitones)")
plt.title(f"Recognition success vs. pitch shift — query: '{QUERY_SONG}'")
plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("figs/06_pitch_robustness.png", dpi=120); plt.close()

# 7. Single-peak vs paired-hash comparison (real database, exact clip query)
best_pairs, hist_pairs, _ = match_query(query_x, FS, db, nperseg=4096,
                                         neighborhood=(20, 20), amp_min_db=-60,
                                         use_pairs=True)
best_single, hist_single, _ = match_query(query_x, FS, db, nperseg=4096,
                                           neighborhood=(20, 20), amp_min_db=-60,
                                           use_pairs=False)
sid_true = [sid for sid, n in db.song_names.items() if n == QUERY_SONG][0]
top_pairs = hist_pairs[sid_true].most_common(3)
top_single = hist_single[sid_true].most_common(3) if best_single is not None else []
print()
print(f"Paired-hash match  -> predicted: {best_pairs}, top offsets/counts: {top_pairs}")
print(f"Single-peak match  -> predicted: {best_single}, top offsets/counts: {top_single}")

print("All figures saved to ./figs/")
for fn in sorted(os.listdir("figs")):
    print(" -", fn)
