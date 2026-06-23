# EE200 Q3 — Audio Fingerprinting (Shazam clone)

**Status: tested end-to-end on the real 50-song database provided for this
project.** All 50 songs index successfully, exact and noisy/clipped queries
match correctly, and the noise/pitch-shift robustness experiments produced
the real numbers quoted in `Q3A_report.md`.

**`app.py` has been rebuilt to match the professor's demo video**: dark
teal/orange theme, Library / Identify / Batch tabs, a per-stage timing
strip, a "match found" card with cluster-score-vs-runner-up margin, and a
3-step narrated result (spectrogram+constellation, full-song fingerprint
with the matched window highlighted, and the offset "alignment spike").
All custom visuals were tested standalone in this sandbox (see `figs/` and
the description below) and render correctly — **but Streamlit itself is
not installed in this sandbox, so the full app has not been run in an
actual browser.** Run `streamlit run app.py` locally as your first check
before deploying.

## Files
- `fingerprint.py` — core library: spectrogram, peak-picking (constellation),
  pairwise hashing, database build, and offset-histogram matching. Audio
  loading uses the `ffmpeg` binary directly (via subprocess) so MP3s decode
  without needing `soundfile`/`librosa` installed. Also includes
  `match_query_detailed()` (per-stage timing + candidate list + confidence
  margin, used by the new UI) and `get_full_song_fingerprint()` /
  `song_accent_color()` helpers for the Library grid and Step 2 visual.
- `self_test.py` — validates the whole pipeline on synthetic signals
  (no real audio needed, useful as a fast logic check).
- `q3a_analysis.py` — generates every plot needed for the Q3A report into
  `figs/`, using the real provided songs (DFT vs spectrogram, window-length
  comparison, constellation, noise/pitch robustness curves, pairs-vs-singles
  comparison).
- `Q3A_report.md` — written report/discussion for Q3A, with real measured
  numbers from the provided dataset.
- `app.py` — Streamlit app for Q3B (Library / Identify / Batch tabs),
  styled to match the demo video.
- `packages.txt` — tells Streamlit Community Cloud to install `ffmpeg`
  (needed for MP3 decoding in the deployed environment).
- `requirements.txt` — Python dependencies (numpy/scipy/matplotlib/
  streamlit/pillow).

## Before you deploy: run it locally first

```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL it prints and click through all three tabs. Things to
specifically check, since I could not verify these myself without a
browser:
- The custom CSS renders as intended (fonts, colors, card borders) — some
  CSS selectors targeting Streamlit's internal DOM (e.g. `data-baseweb`
  attributes) can change between Streamlit versions and may need small
  tweaks.
- The Library tab loads in a few seconds, not tens of seconds (the
  expensive full-database scan is cached, so it should only be slow once).
- Uploading a real query clip on the Identify tab produces all 3 steps
  without error.
- The Batch tab produces a `results.csv` with exactly two columns.

## Adding "try a sample" clips (shown in the demo)

The demo's Identify tab offers 5 pre-loaded sample clips with inline
players and "Try" buttons. To reproduce this, create a `samples/` folder
next to `app.py` containing a few short (~10-30s) excerpts cut from your
song library (e.g. with `ffmpeg -i song.mp3 -ss 30 -t 15 samples/sample1.mp3`).
`app.py` automatically detects and lists any audio files placed there.

## Using the real song dataset

The song folder is expected at:
```
songs/EE200 Project Song Database/<Song Name>.mp3
```
(rename/move your unzipped folder to `songs/` at the repo root, or change
`SONG_FOLDER` in `app.py` / `q3a_analysis.py` to match wherever you place it).

Build the database once:
```python
from fingerprint import build_database
build_database("songs/EE200 Project Song Database", db_path="song_db.pkl")
```
This writes `song_db.pkl`, which both the notebook and `app.py` reuse
(no need to re-index every run). **Indexing all 50 songs takes about a
minute** on a typical machine. The resulting `song_db.pkl` is ~56 MB with
default settings (`fan_out=3`) — comfortably under GitHub's 100 MB
per-file limit, so it can be committed directly without Git LFS.

3. For the Q3A notebook, replace `make_song(...)` calls in
   `q3a_analysis.py` with:
   ```python
   from fingerprint import load_audio
   x, fs = load_audio("songs/<a_provided_song>.wav")
   ```
   and re-run — every plot (DFT, spectrogram, constellation, robustness
   curves) regenerates from real audio automatically.

4. For matching a query clip:
   ```python
   from fingerprint import FingerprintDB, match_query, load_audio
   db = FingerprintDB.load("song_db.pkl")
   x, fs = load_audio("query_clip.wav")
   best_name, offset_hist, qres = match_query(x, fs, db)
   print("Predicted song:", best_name)
   ```

## Single peaks vs. paired hashes

`match_query(..., use_pairs=True)` (default) uses the real two-peak hash.
Pass `use_pairs=False` to run the single-peak variant for the comparison
the assignment asks for.

## Noise / pitch-shift experiments

Helper functions in `self_test.py`:
```python
from self_test import add_noise, pitch_shift_resample
noisy_query    = add_noise(x, snr_db=10)
shifted_query  = pitch_shift_resample(x, fs, semitones=2)
```
Run these through `match_query` at increasing noise/shift levels and plot
success vs. parameter, exactly as `q3a_analysis.py` does.

## Deploying the Q3B app (Streamlit Community Cloud)

1. Push this whole folder to a **public GitHub repo**, including:
   - `app.py`, `fingerprint.py`
   - `song_db.pkl` (prebuilt database — so the app works immediately,
     without re-indexing on every cold start)
   - `requirements.txt` (see below)
2. Go to https://share.streamlit.io → "New app" → connect the repo → set
   main file to `app.py` → Deploy.
3. Once live, copy the public URL into your Q3 PDF, along with a link to
   the GitHub source.
4. Also zip all code (`fingerprint.py`, `app.py`, notebook, `song_db.pkl`)
   and submit per the assignment's instructions.

### requirements.txt (create this alongside app.py before deploying)
```
numpy
scipy
matplotlib
streamlit
soundfile
librosa
```

## Batch mode CSV format (graded automatically — must match exactly)
```
filename,prediction
query1.wav,song_title_without_extension
query2.mp3,another_song
```
`app.py`'s batch mode already produces this format via the download button.
