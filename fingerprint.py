"""
fingerprint.py
--------------
Core Shazam-style audio fingerprinting library for EE200 Q3.

Pipeline:
    audio -> spectrogram -> constellation (local peaks) -> hashes (f1, f2, dt)
    -> database lookup -> offset histogram -> best match

No external audio libraries are required for the ALGORITHM itself (only
numpy/scipy/matplotlib). Loading real song files (mp3/wav) uses
`load_audio()`, which tries soundfile/librosa if available and falls back
to scipy.io.wavfile for plain .wav files.
"""

import os
import glob
import pickle
import shutil
import subprocess
import tempfile
import numpy as np
from scipy import signal
from scipy.io import wavfile as _wavfile
from scipy.ndimage import maximum_filter, binary_erosion, generate_binary_structure
from collections import defaultdict, Counter

# ----------------------------------------------------------------------
# 1. Audio loading
# ----------------------------------------------------------------------

def _load_via_ffmpeg(path, target_sr):
    """
    Decode ANY audio format (mp3, m4a, flac, ...) to mono PCM using the
    ffmpeg binary directly. This avoids needing soundfile/librosa installed
    -- only the `ffmpeg` executable needs to be on PATH (it already is on
    most systems, including Streamlit Community Cloud once added via
    packages.txt).
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", path,
            "-ac", "1",                # mono
            "-ar", str(target_sr),     # resample directly to target rate
            "-f", "wav",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='ignore')}")

        sr, x = _wavfile.read(tmp_path)
        x = x.astype(np.float64)
        if x.dtype != np.float64:
            pass
        # normalize integer PCM to [-1, 1]
        info_max = np.iinfo(np.int16).max if x.dtype == np.int16 else None
        return x, sr
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def load_audio(path, target_sr=11025, mono=True):
    """
    Load an audio file and return (signal, sample_rate).
    Downsamples to target_sr (11025 Hz is plenty for fingerprinting;
    speech/music identity lives mostly below ~5 kHz).

    Tries, in order: soundfile -> librosa -> ffmpeg binary -> scipy wavfile.
    The ffmpeg fallback means mp3/m4a/etc. work even without
    soundfile/librosa installed, as long as the `ffmpeg` executable exists.
    """
    sr = None
    x = None

    # Try soundfile first (handles wav, flac, ogg; mp3 on some platforms)
    try:
        import soundfile as sf
        x, sr = sf.read(path, always_2d=False)
    except Exception:
        pass

    # Fall back to librosa (handles mp3 robustly via audioread/ffmpeg)
    if x is None:
        try:
            import librosa
            x, sr = librosa.load(path, sr=None, mono=mono)
        except Exception:
            pass

    # Fall back to calling the ffmpeg binary directly (decodes to WAV,
    # already resampled to target_sr -- skips the resample step below)
    if x is None:
        try:
            x, sr = _load_via_ffmpeg(path, target_sr)
            x = np.asarray(x, dtype=np.float64)
            if x.ndim > 1 and mono:
                x = x.mean(axis=1)
            peak = np.max(np.abs(x)) + 1e-12
            x = x / peak
            return x, sr  # already at target_sr, skip resample/normalize below
        except Exception:
            pass

    # Last resort: plain wav via scipy
    if x is None:
        from scipy.io import wavfile
        sr, x = wavfile.read(path)
        x = x.astype(np.float32)
        if x.dtype.kind in ("i", "u"):
            x = x / np.iinfo(x.dtype).max

    if x is None:
        raise IOError(f"Could not load audio file: {path}")

    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1 and mono:
        x = x.mean(axis=1)

    if sr != target_sr:
        num_samples = int(len(x) * target_sr / sr)
        x = signal.resample(x, num_samples)
        sr = target_sr

    # normalize amplitude to [-1, 1]
    peak = np.max(np.abs(x)) + 1e-12
    x = x / peak
    return x, sr


# ----------------------------------------------------------------------
# 2. Spectrogram
# ----------------------------------------------------------------------

def compute_spectrogram(x, fs, nperseg=4096, noverlap=None, window="hann"):
    """
    Compute the magnitude spectrogram (in dB) of signal x.
    Returns f (Hz), t (s), Sxx_db (freq_bins x time_bins).
    """
    if noverlap is None:
        noverlap = nperseg * 3 // 4  # 75% overlap -> good time resolution
    f, t, Sxx = signal.spectrogram(
        x, fs=fs, window=window, nperseg=nperseg,
        noverlap=noverlap, mode="magnitude"
    )
    Sxx_db = 20 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


# ----------------------------------------------------------------------
# 3. Constellation map: local-maxima peak picking
# ----------------------------------------------------------------------

def find_peaks_2d(Sxx_db, amp_min_db=-60, neighborhood=(20, 20)):
    """
    Find local maxima in the spectrogram that stand out from their
    surroundings -- the 'constellation' points.

    neighborhood: (freq_size, time_size) of the local max filter window.
    amp_min_db:   ignore peaks quieter than this (after dB normalization,
                  loudest point in the spectrogram = 0 dB).
    Returns: list of (freq_bin_idx, time_bin_idx) peak coordinates.
    """
    # normalize so loudest point is 0 dB
    Sxx_norm = Sxx_db - Sxx_db.max()

    struct = np.ones(neighborhood, dtype=bool)
    local_max = maximum_filter(Sxx_norm, footprint=struct) == Sxx_norm

    # remove peaks that are too quiet
    loud_enough = Sxx_norm > amp_min_db
    peak_mask = local_max & loud_enough

    freq_idx, time_idx = np.nonzero(peak_mask)
    peaks = list(zip(freq_idx, time_idx))
    return peaks


# ----------------------------------------------------------------------
# 4. Hashing: pair nearby peaks
# ----------------------------------------------------------------------

FAN_OUT = 3          # how many neighbors to pair each anchor peak with
MIN_DT_BINS = 1       # minimum time-bin gap between paired peaks
MAX_DT_BINS = 60       # maximum time-bin gap between paired peaks


def generate_hashes(peaks, fan_out=FAN_OUT, min_dt=MIN_DT_BINS, max_dt=MAX_DT_BINS):
    """
    Pair each peak (anchor) with up to `fan_out` nearby peaks that occur
    later in time, within [min_dt, max_dt] time bins.

    Returns: list of (hash_key, anchor_time_bin)
        hash_key = (freq1_bin, freq2_bin, dt_bins)
    """
    peaks_sorted = sorted(peaks, key=lambda p: p[1])  # sort by time bin
    n = len(peaks_sorted)
    hashes = []

    for i in range(n):
        f1, t1 = peaks_sorted[i]
        count = 0
        for j in range(i + 1, n):
            f2, t2 = peaks_sorted[j]
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break  # sorted by time, no point looking further
            hash_key = (f1, f2, dt)
            hashes.append((hash_key, t1))
            count += 1
            if count >= fan_out:
                break
    return hashes


# ----------------------------------------------------------------------
# 5. Database build / single-peak variant
# ----------------------------------------------------------------------

def fingerprint_signal(x, fs, nperseg=4096, noverlap=None,
                        amp_min_db=-60, neighborhood=(20, 20),
                        fan_out=FAN_OUT):
    """
    Full pipeline: signal -> spectrogram -> peaks -> hashes.
    Returns dict with intermediate results for plotting + the hash list.
    """
    f, t, Sxx_db = compute_spectrogram(x, fs, nperseg=nperseg, noverlap=noverlap)
    peaks = find_peaks_2d(Sxx_db, amp_min_db=amp_min_db, neighborhood=neighborhood)
    hashes = generate_hashes(peaks, fan_out=fan_out)
    return {
        "f": f, "t": t, "Sxx_db": Sxx_db,
        "peaks": peaks, "hashes": hashes,
    }


class FingerprintDB:
    """
    Maps hash_key -> list of (song_id, anchor_time_bin).
    Also stores song_id -> filename and the dt-per-bin (for offset -> seconds).
    """
    def __init__(self):
        self.hash_table = defaultdict(list)
        self.song_names = {}     # song_id -> filename (no extension)
        self.time_per_bin = None # seconds per time-bin (set when first song is added)

    def add_song(self, song_id, name, hashes, t_axis):
        self.song_names[song_id] = name
        if self.time_per_bin is None and len(t_axis) > 1:
            self.time_per_bin = t_axis[1] - t_axis[0]
        for hash_key, anchor_t in hashes:
            self.hash_table[hash_key].append((song_id, anchor_t))

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump({
                "hash_table": dict(self.hash_table),
                "song_names": self.song_names,
                "time_per_bin": self.time_per_bin,
            }, fh)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        db = cls()
        db.hash_table = defaultdict(list, data["hash_table"])
        db.song_names = data["song_names"]
        db.time_per_bin = data["time_per_bin"]
        return db


def build_database(song_folder, db_path="song_db.pkl",
                    nperseg=4096, noverlap=None,
                    amp_min_db=-60, neighborhood=(20, 20), fan_out=FAN_OUT,
                    extensions=(".wav", ".mp3", ".flac")):
    """
    Index every song in `song_folder` into a FingerprintDB and save it.
    The song's *filename without extension* becomes its label, per the
    assignment's submission rules.
    """
    db = FingerprintDB()
    files = []
    for ext in extensions:
        files.extend(sorted(glob.glob(os.path.join(song_folder, f"*{ext}"))))

    for song_id, path in enumerate(files):
        name = os.path.splitext(os.path.basename(path))[0]
        x, fs = load_audio(path)
        result = fingerprint_signal(x, fs, nperseg=nperseg, noverlap=noverlap,
                                     amp_min_db=amp_min_db,
                                     neighborhood=neighborhood, fan_out=fan_out)
        db.add_song(song_id, name, result["hashes"], result["t"])
        print(f"Indexed [{song_id}] {name}: {len(result['hashes'])} hashes")

    db.save(db_path)
    return db


# ----------------------------------------------------------------------
# 6. Matching a query clip
# ----------------------------------------------------------------------

def match_query_detailed(x, fs, db: FingerprintDB,
                          nperseg=4096, noverlap=None,
                          amp_min_db=-60, neighborhood=(20, 20), fan_out=FAN_OUT,
                          confidence_threshold=10):
    """
    Richer version of match_query() that returns everything the UI's
    step-by-step narrative needs in one call, with per-stage timing.

    Returns a dict:
        best_name, best_song_id, cluster_score, runner_up_score, margin_x,
        candidates: [(name, score), ...] sorted desc (top 5),
        offset_hist (Counter for the best song),
        f, t, Sxx_db, peaks, hashes  (query spectrogram/constellation),
        best_offset (int, frames),
        timings: dict(stage -> milliseconds), total_ms
        is_confident: bool (score / runner_up >= confidence_threshold, or
                             no runner-up at all)
    """
    import time
    timings = {}

    t0 = time.perf_counter()
    f, t, Sxx_db = compute_spectrogram(x, fs, nperseg=nperseg, noverlap=noverlap)
    timings["spectrogram"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    peaks = find_peaks_2d(Sxx_db, amp_min_db=amp_min_db, neighborhood=neighborhood)
    timings["constellation"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    hashes = generate_hashes(peaks, fan_out=fan_out)
    timings["hashing"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    offset_histograms = defaultdict(Counter)
    for hash_key, query_t in hashes:
        if hash_key in db.hash_table:
            for song_id, db_t in db.hash_table[hash_key]:
                offset = db_t - query_t
                offset_histograms[song_id][offset] += 1
    timings["db_lookup"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scores = {}
    for song_id, hist in offset_histograms.items():
        if hist:
            _, best_count = hist.most_common(1)[0]
            scores[song_id] = best_count
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    timings["scoring"] = (time.perf_counter() - t0) * 1000
    timings["total"] = sum(timings.values())

    result = {
        "f": f, "t": t, "Sxx_db": Sxx_db, "peaks": peaks, "hashes": hashes,
        "timings": timings, "candidates": [], "best_name": None,
        "best_song_id": None, "cluster_score": 0, "runner_up_score": 0,
        "margin_x": None, "offset_hist": Counter(), "best_offset": None,
        "is_confident": False,
    }

    if not ranked:
        return result

    best_song_id, cluster_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0
    margin_x = (cluster_score / runner_up_score) if runner_up_score > 0 else None

    result.update({
        "best_name": db.song_names[best_song_id],
        "best_song_id": best_song_id,
        "cluster_score": cluster_score,
        "runner_up_score": runner_up_score,
        "margin_x": margin_x,
        "candidates": [(db.song_names[sid], sc) for sid, sc in ranked[:5]],
        "offset_hist": offset_histograms[best_song_id],
        "best_offset": offset_histograms[best_song_id].most_common(1)[0][0],
        "is_confident": (margin_x is None) or (margin_x >= confidence_threshold),
    })
    return result


def get_full_song_fingerprint(db: FingerprintDB, song_id):
    """
    Reconstruct the full set of (time_bin, freq_bin) anchor points stored
    for one song, for the 'Step 2: where in the song' visualization.
    Returns (times, freqs) arrays.
    """
    times, freqs = [], []
    for (f1, f2, dt), entries in db.hash_table.items():
        for sid, t in entries:
            if sid == song_id:
                times.append(t)
                freqs.append(f1)
    return np.array(times), np.array(freqs)


def make_constellation_thumbnail(peaks, f_bins, t_bins, color="#2dd4bf",
                                  size=(220, 140)):
    """
    Render a small constellation scatter as a base64 PNG data-URI, for use
    as a Library-tab thumbnail card. Avoids matplotlib for speed; draws
    directly with PIL.
    """
    import io
    import base64
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, color="#0a1416")
    draw = ImageDraw.Draw(img)
    if len(peaks) == 0:
        buf = io.BytesIO(); img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    fs = np.array([p[0] for p in peaks], dtype=float)
    ts = np.array([p[1] for p in peaks], dtype=float)
    fs_norm = 1 - (fs / max(f_bins, 1))   # invert: high freq -> top
    ts_norm = ts / max(t_bins, 1)

    rgb = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    for x_n, y_n in zip(ts_norm, fs_norm):
        x_px = int(x_n * (size[0] - 4)) + 2
        y_px = int(y_n * (size[1] - 4)) + 2
        draw.ellipse([x_px-1, y_px-1, x_px+1, y_px+1], fill=rgb)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def song_accent_color(name):
    """Deterministic accent color per song (for thumbnail tinting), based
    on a simple hash of the name -- so colors stay stable across runs."""
    palette = ["#2dd4bf", "#f59e0b", "#a78bfa", "#f472b6", "#60a5fa",
               "#34d399", "#fb923c", "#c084fc"]
    h = sum(ord(c) for c in name)
    return palette[h % len(palette)]


def display_name(name):
    """
    Cosmetic-only fix: the provided song files use underscores where the
    original titles had apostrophes (e.g. 'Don_t Stop Me Now'), because
    apostrophes aren't safe in filenames. We never rename the actual files
    or change the value used for matching/labels -- this only prettifies
    what's shown on screen, by turning '_' into an apostrophe when it sits
    between a letter and a common contraction ending.
    """
    import re
    return re.sub(r"(?<=[A-Za-z])_(?=(s|t|ll|ve|re|d|m)\b)", "'", name)


def match_query(x, fs, db: FingerprintDB,
                 nperseg=4096, noverlap=None,
                 amp_min_db=-60, neighborhood=(20, 20), fan_out=FAN_OUT,
                 use_pairs=True, top_k=1):
    """
    Fingerprint a query signal and match it against the database.

    If use_pairs=True: uses paired (f1, f2, dt) hashes (the real algorithm).
    If use_pairs=False: uses single-peak (freq_bin) matching for comparison
        (weaker discriminative power, kept for the assignment's
        "single peaks vs pairs" experiment).

    Returns: best_song_name, offset_histograms (dict song_id -> Counter),
             query_result (spectrogram/peaks/hashes, for plotting)
    """
    query_result = fingerprint_signal(x, fs, nperseg=nperseg, noverlap=noverlap,
                                       amp_min_db=amp_min_db,
                                       neighborhood=neighborhood, fan_out=fan_out)

    offset_histograms = defaultdict(Counter)

    if use_pairs:
        for hash_key, query_t in query_result["hashes"]:
            if hash_key in db.hash_table:
                for song_id, db_t in db.hash_table[hash_key]:
                    offset = db_t - query_t
                    offset_histograms[song_id][offset] += 1
    else:
        # single-peak variant: build a lookup of freq_bin -> [(song_id, t)]
        single_table = defaultdict(list)
        for hash_key, db_t in [(hk, t) for hk, lst in db.hash_table.items()
                                for (sid, t) in lst]:
            pass  # placeholder, replaced below for clarity/perf

        # Build a simple single-frequency index directly from peaks instead
        # (re-derive per-song single-peak table once, cached on db object)
        if not hasattr(db, "_single_peak_table"):
            db._single_peak_table = defaultdict(list)
            # Reconstruct from hash_table anchors' f1 component
            for (f1, f2, dt), entries in db.hash_table.items():
                for (song_id, t) in entries:
                    db._single_peak_table[f1].append((song_id, t))

        for (f, t_bin) in query_result["peaks"]:
            for song_id, db_t in db._single_peak_table.get(f, []):
                offset = db_t - t_bin
                offset_histograms[song_id][offset] += 1

    # score each song by the size of its largest offset "spike"
    scores = {}
    for song_id, hist in offset_histograms.items():
        if hist:
            best_offset, best_count = hist.most_common(1)[0]
            scores[song_id] = best_count

    if not scores:
        return None, offset_histograms, query_result

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_song_id = ranked[0][0]
    best_name = db.song_names[best_song_id]

    if top_k > 1:
        top_names = [(db.song_names[sid], sc) for sid, sc in ranked[:top_k]]
        return best_name, offset_histograms, query_result, top_names

    return best_name, offset_histograms, query_result
