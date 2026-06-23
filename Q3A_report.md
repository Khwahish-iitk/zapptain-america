# Q3A — Sonic Signatures: Report

*(Figures referenced below are in `figs/`, generated from the **real provided
song database** — 50 songs, e.g. "Yesterday" used as the main analysis/query
example. Database built once with `build_database()` over all 50 tracks and
saved to `song_db.pkl`. Re-run `q3a_analysis.py` to regenerate everything
from scratch if parameters change.)*

## 1. Why a single DFT is not enough

The whole-clip DFT (`figs/01_full_dft.png`) tells us *which* frequencies are
present, but a magnitude spectrum collapses the entire time axis into a
single number per frequency bin — there is no way to recover *when* any of
those frequencies occurred. Two completely different songs that happen to
contain the same set of notes, played in a different order, can produce
nearly identical DFT magnitude plots. Since identifying a song means
recognizing a specific *sequence* of sounds over time, we need a
representation that keeps both axes.

## 2. The spectrogram

The spectrogram (`figs/02_spectrogram_main.png`) is built by sliding a
window of length `nperseg` along the signal, taking the DFT of each
windowed segment, and stacking the magnitudes as columns indexed by time.
Each chirp component in the test signal traces a clean diagonal streak, and
each steady tone would trace a horizontal line — exactly matching the
qualitative description in the assignment.

## 3. Window-length trade-off

Comparing `figs/03_spectrogram_short_256.png` and
`figs/03_spectrogram_long_8192.png`:

- **Short window (256 samples ≈ 23 ms):** good *time* resolution — onsets
  and fast transients are localized sharply — but poor *frequency*
  resolution, so the rising/falling chirps appear as thick, blurry bands.
- **Long window (8192 samples ≈ 743 ms):** good *frequency* resolution —
  narrow, sharply defined frequency lines — but poor *time* resolution, so
  fast frequency changes get smeared horizontally and brief events blur
  together.

This is the time–frequency uncertainty trade-off (the STFT analogue of the
windowing trade-off seen in Q2c/Q2e): you cannot simultaneously have
arbitrarily fine resolution in both axes. A practical fingerprinting system
picks a middle-ground window (e.g. 1024–4096 samples at typical audio
sample rates) that resolves both note onsets and pitch well enough for
robust peak-picking.

## 4. Constellation map

Local 2-D maxima of the spectrogram — points that are louder than every
neighbor within a small time–frequency neighborhood — are kept as the
fingerprint "stars" (`figs/04_constellation.png`). These are the most
energetic, most reproducible points: even after compression, noise, or
channel distortion, the loudest local peaks tend to survive, while the
exact dB value of quieter background content does not.

## 5. Hashing: pairs vs. single peaks, and why pairs win

A single peak is just `(frequency, time)`. Across thousands of songs, any
one frequency bin will appear constantly — collisions are essentially
guaranteed, so a single-peak match carries very little discriminating
power. Tested directly on the real database (10s clip of "Yesterday" vs.
all 50 songs):

| Method | Top offset count (true song) | Runner-up offset count (false song) | Margin |
|---|---|---|---|
| Paired hashes `(f1,f2,Δt)` | 1081 | 3 | ~360× |
| Single peaks `(f)` | 1110 | 27 | ~41× |

Both methods correctly pick the true song here, but the **margin of
confidence is over 20× sharper with paired hashes**. The single-peak
runner-up count (72) is far closer to contaminating the decision than the
paired-hash runner-up (3) — on a harder query (more noise, a shorter clip,
or a database with more similar-sounding songs), that weaker margin is
exactly what would flip a single-peak system to a wrong answer while the
paired-hash system still gets it right.

Pairing an anchor peak with a handful of nearby peaks into a hash
`(f1, f2, Δt)` is far more specific: it encodes a small *local pattern* of
the spectrogram rather than an isolated point. The combinatorics make a
coincidental collision between two unrelated songs' hash tables vastly
less likely, because the pair must agree in *three* numbers (`f1`, `f2`,
and the time gap) simultaneously, not just one. The deciding evidence is
the **offset histogram**: for a true match, the database time and query
time of every matching hash differ by the *same* constant offset (the
position of the query within the song), so all matches stack into a single
sharp spike. For an unrelated song, hash collisions occur at essentially
random offsets, since there's no consistent alignment — producing a flat,
scattered histogram. Pairs make that spike much taller relative to the
noise floor than single peaks do.

## 6. Robustness experiments

**Noise (`figs/05_noise_robustness.png`):** Testing a 10-second clip of
"Yesterday" (taken from 30s–40s into the track) against the full 50-song
database: recognition is **perfectly correct down to −5 dB SNR** (noise
power exceeding signal power by 5 dB!) and only fails at **−10 dB and
below**. Real music tolerates noise far better than a simple synthetic
tone, because a real song's spectrogram has many redundant strong peaks
(vocals, harmonics, multiple instruments) — losing some peaks to noise
still leaves plenty of surviving hashes that vote for the correct offset.
The failure is fairly abrupt rather than gradual, because the matching
decision itself is a hard threshold (best song = most hash collisions at
one offset) — once noise corrupts enough of the true peaks, the genuine
offset spike falls below a spurious one from another song.

**Pitch shift / time stretch (`figs/06_pitch_robustness.png`):** In sharp
contrast, even a **0.25-semitone pitch shift broke the matcher** on the
same clip (it returned a completely unrelated song with high confidence).
This is because a pitch shift moves *every* frequency component by the same
multiplicative factor — shifting frequency bins just slightly is usually
enough to move a peak into a different discrete frequency bin (or off the
hash's exact `(f1, f2)` key entirely), and an exact-match hash lookup
requires bit-for-bit identical keys. To a human ear the song is
unmistakably the same (our pitch perception is forgiving of small relative
shifts), but to a table lookup keyed on exact bin numbers, even one bin of
drift means zero hash collisions. This is a striking illustration of how
fragile naive exact-hash matching is to even imperceptible distortions —
**far more fragile than to substantial added noise**.

**Suggested robustness improvement:** instead of requiring an *exact* hash
match, allow a small tolerance window on the frequency bins (e.g., match if
`f1`, `f2` are within ±1–2 bins) or, better, hash *frequency ratios*
`f2/f1` rather than absolute frequencies — ratios are invariant to a
uniform pitch shift, since multiplying every frequency by a constant factor
leaves their ratio unchanged. (Commercial systems such as Shazam use
several such tricks, including combining multiple hash variants and
relying on the sheer number of independent hashes so a few survive a
moderate distortion.)

## 7. Summary

| Stage | What it does | Why it matters |
|---|---|---|
| Spectrogram | Time-localized DFT | Captures *when* each frequency occurs |
| Constellation (peak-picking) | Keep only strong local maxima | Sparse, noise-resistant fingerprint |
| Pairwise hashing | `(f1, f2, Δt)` keys | Sharp discriminating power, few false collisions |
| Offset histogram | Vote for consistent time alignment | Separates true match from coincidental collisions |
