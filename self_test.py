"""
self_test.py — sanity-checks fingerprint.py using synthetic "songs"
(sums of chirps/tones), since no real audio files are available in
this sandbox. Run this to confirm the algorithm logic is correct
before pointing it at the real song dataset.
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from fingerprint import (fingerprint_signal, FingerprintDB, match_query,
                          compute_spectrogram, find_peaks_2d, generate_hashes)

FS = 11025
DUR = 20  # seconds per "song"

def make_song(seed, duration=DUR, fs=FS):
    """Create a synthetic song: a handful of tones/chirps with changing
    frequencies over time, so it has real time-frequency structure
    (unlike a single pure tone, which would be a trivial test)."""
    rng = np.random.RandomState(seed)
    t = np.arange(int(duration * fs)) / fs
    x = np.zeros_like(t)
    n_components = 6
    for _ in range(n_components):
        f_start = rng.uniform(200, 3000)
        f_end = f_start + rng.uniform(-500, 500)
        amp = rng.uniform(0.3, 1.0)
        # simple linear chirp component
        phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration))
        x += amp * np.sin(phase)
    x = x / np.max(np.abs(x))
    return x

def add_noise(x, snr_db):
    sig_power = np.mean(x**2)
    noise_power = sig_power / (10**(snr_db / 10))
    noise = np.random.RandomState(0).normal(0, np.sqrt(noise_power), size=x.shape)
    return x + noise

def pitch_shift_resample(x, fs, semitones):
    """Crude pitch shift: resample then it changes both pitch AND speed
    (good enough to demonstrate the fingerprint-breaking effect)."""
    factor = 2 ** (semitones / 12)
    n_new = int(len(x) / factor)
    from scipy.signal import resample
    return resample(x, n_new)

def main():
    print("=" * 60)
    print("STEP 1: Build a tiny synthetic song database (5 songs)")
    print("=" * 60)
    songs = {f"song_{i:02d}": make_song(seed=i) for i in range(5)}

    db = FingerprintDB()
    song_results = {}
    for sid, (name, x) in enumerate(songs.items()):
        result = fingerprint_signal(x, FS, nperseg=2048, neighborhood=(15, 15),
                                     amp_min_db=-40)
        db.add_song(sid, name, result["hashes"], result["t"])
        song_results[name] = result
        print(f"  [{sid}] {name}: {len(result['peaks'])} peaks, "
              f"{len(result['hashes'])} hashes")

    print()
    print("=" * 60)
    print("STEP 2: Exact-clip query (should match perfectly, offset=0)")
    print("=" * 60)
    query_name = "song_02"
    query_x = songs[query_name][: 5 * FS]  # first 5 seconds as the "query"
    best, hist, qres = match_query(query_x, FS, db, nperseg=2048,
                                    neighborhood=(15, 15), amp_min_db=-40)
    print(f"  True song: {query_name}  |  Predicted: {best}  ->",
          "PASS" if best == query_name else "FAIL")

    print()
    print("=" * 60)
    print("STEP 3: Noise robustness — increasing noise until it breaks")
    print("=" * 60)
    for snr in [30, 20, 10, 5, 0, -5]:
        noisy = add_noise(query_x, snr)
        best, hist, qres = match_query(noisy, FS, db, nperseg=2048,
                                        neighborhood=(15, 15), amp_min_db=-40)
        status = "PASS" if best == query_name else f"FAIL (got {best})"
        print(f"  SNR={snr:>4} dB  ->  predicted: {best!s:<10}  {status}")

    print()
    print("=" * 60)
    print("STEP 4: Pitch shift / time stretch — should break matching")
    print("=" * 60)
    for semitones in [0, 0.5, 1, 2, 4]:
        shifted = pitch_shift_resample(query_x, FS, semitones)
        best, hist, qres = match_query(shifted, FS, db, nperseg=2048,
                                        neighborhood=(15, 15), amp_min_db=-40)
        status = "PASS" if best == query_name else f"FAIL (got {best})"
        print(f"  shift={semitones:>4} semitones  ->  predicted: {best!s:<10}  {status}")

    print()
    print("=" * 60)
    print("STEP 5: Single-peak matching vs paired-hash matching")
    print("=" * 60)
    best_pairs, hist_pairs, _ = match_query(query_x, FS, db, nperseg=2048,
                                             neighborhood=(15, 15), amp_min_db=-40,
                                             use_pairs=True)
    best_single, hist_single, _ = match_query(query_x, FS, db, nperseg=2048,
                                               neighborhood=(15, 15), amp_min_db=-40,
                                               use_pairs=False)
    top_pairs = hist_pairs[ [sid for sid,n in db.song_names.items() if n==best_pairs][0] ].most_common(1)
    print(f"  Paired hashes  -> predicted: {best_pairs}, top-offset count: {top_pairs}")
    if best_single is not None:
        sid_single = [sid for sid, n in db.song_names.items() if n == best_single][0]
        top_single = hist_single[sid_single].most_common(1)
        print(f"  Single peaks   -> predicted: {best_single}, top-offset count: {top_single}")
    print("  (Pairs should give a sharper, more dominant peak count than single peaks.)")

if __name__ == "__main__":
    main()
