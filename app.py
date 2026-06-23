"""
app.py — Streamlit app for EE200 Q3B: 'Zapp tain America'

Rebuilt to match the professor's demo video:
  - LIBRARY tab: grid of song cards, each with a constellation thumbnail
    and hash count.
  - IDENTIFY tab: upload or try a sample clip -> step-by-step narrated
    results (pipeline timing, match card, candidate scores, Step 1
    feature extraction, Step 2 database search, Step 3 the alignment
    spike / offset histogram proof).
  - BATCH tab: upload many clips -> results.csv with filename,prediction.

Run locally with:   streamlit run app.py
Deploy on Streamlit Community Cloud (push this repo, including
song_db.pkl and the songs/ folder, to GitHub, then connect it there).
"""
import os
import io
import csv
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

from fingerprint import (
    load_audio, fingerprint_signal, FingerprintDB, build_database,
    match_query_detailed, get_full_song_fingerprint,
    make_constellation_thumbnail, song_accent_color,
)

DB_PATH = "song_db.pkl"
SONG_FOLDER = "songs/EE200 Project Song Database"
SAMPLES_FOLDER = "samples"  # optional: a few short pre-clipped query mp3s for "try a sample"
CONFIDENCE_THRESHOLD = 10   # cluster_score must be >= this many x the runner-up

# ----------------------------------------------------------------------
# Color tokens (matched to the demo video)
# ----------------------------------------------------------------------
BG = "#000000"
PANEL = "#0c1013"
PANEL_BORDER = "#1a2126"
CARD_BORDER = "#18211e"
TEAL = "#2dd4bf"
TEAL_DIM = "#17795d"
ORANGE = "#f89e22"
TEXT = "#e8edec"
TEXT_MUTED = "#7c8a89"
TEXT_FAINT = "#4a5654"

st.set_page_config(page_title="Zapptai America", layout="wide",
                    initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------
# Global CSS — dark mono theme
# ----------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {{
    background: {BG};
    color: {TEXT};
}}
section[data-testid="stSidebar"] {{ display: none; }}
.block-container {{ padding-top: 2rem; max-width: 1100px; }}

* {{ font-family: 'Inter', sans-serif; }}
.mono {{ font-family: 'JetBrains Mono', monospace; }}

.eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {TEXT_FAINT};
    margin-bottom: 4px;
}}

.app-header {{ display:flex; align-items:center; gap:14px; margin-bottom:2px; }}
.app-icon {{
    width:44px; height:44px; border-radius:10px;
    border:1px solid {TEAL_DIM}; display:flex; align-items:center;
    justify-content:center; background: rgba(45,212,191,0.06);
}}
.app-title {{ font-size:28px; font-weight:800; color:{TEXT}; margin:0; }}
.app-title .accent {{ color:{TEAL}; }}
.app-sub {{ font-family:'JetBrains Mono', monospace; font-size:11px;
    letter-spacing:0.12em; text-transform:uppercase; color:{TEXT_FAINT}; margin-top:2px;}}
.app-tagline {{ color:{TEXT_MUTED}; font-size:14.5px; margin-top:10px; }}

.card {{
    background:{PANEL}; border:1px solid {PANEL_BORDER}; border-radius:10px;
    padding:18px 20px;
}}
.song-card {{
    background:{PANEL}; border:1px solid {CARD_BORDER}; border-radius:10px;
    overflow:hidden; margin-bottom:14px;
}}
.song-card img {{ width:100%; display:block; }}
.song-card .meta {{ padding:10px 12px 12px 12px; }}
.song-name {{ font-size:13.5px; font-weight:600; color:{TEXT};
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.song-hashes {{ font-family:'JetBrains Mono',monospace; font-size:11px;
    color:{TEXT_FAINT}; margin-top:2px; }}

.step-block {{ border-left:2px solid {TEAL}; padding-left:18px; margin: 28px 0 14px 4px; }}
.step-eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:11px;
    letter-spacing:0.12em; text-transform:uppercase; color:{TEAL}; margin-bottom:4px;}}
.step-title {{ font-size:19px; font-weight:700; color:{TEXT}; margin:0 0 8px 0;}}
.step-body {{ color:{TEXT_MUTED}; font-size:13.5px; line-height:1.55; }}
.step-body b {{ color:{TEXT}; }}
.hl-teal {{ color:{TEAL}; font-weight:600; }}
.hl-orange {{ color:{ORANGE}; font-weight:600; }}

.timing-strip {{ display:flex; gap:10px; align-items:stretch; margin: 18px 0; }}
.timing-chip {{ flex:1; background:{PANEL}; border:1px solid {PANEL_BORDER};
    border-radius:8px; padding:10px 12px; text-align:center; }}
.timing-num {{ font-family:'JetBrains Mono',monospace; color:{TEAL};
    font-size:11px; margin-bottom:4px; }}
.timing-label {{ font-family:'JetBrains Mono',monospace; font-size:9.5px;
    letter-spacing:0.08em; text-transform:uppercase; color:{TEXT_FAINT}; }}
.timing-ms {{ font-family:'JetBrains Mono',monospace; font-size:16px;
    color:{TEXT}; font-weight:600; margin:3px 0; }}
.timing-sub {{ font-family:'JetBrains Mono',monospace; font-size:9px; color:{TEXT_FAINT}; }}
.timing-total {{ display:flex; align-items:center; justify-content:flex-end;
    padding-right:6px; font-family:'JetBrains Mono',monospace; color:{TEXT_MUTED}; font-size:13px; min-width:120px;}}

.match-card {{ background: rgba(45,212,191,0.05); border:1px solid {TEAL_DIM};
    border-radius:10px; padding:20px 24px; margin: 16px 0; }}
.match-eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:11px;
    letter-spacing:0.12em; text-transform:uppercase; color:{TEAL}; margin-bottom:6px;}}
.match-title {{ font-size:30px; font-weight:800; color:{TEXT}; margin:0; }}
.match-score {{ font-family:'JetBrains Mono',monospace; font-size:13px;
    color:{TEXT_MUTED}; margin-top:6px; }}
.match-score b {{ color:{ORANGE}; }}

.no-match-card {{ background: rgba(248,113,113,0.05); border:1px solid #7c2d2d;
    border-radius:10px; padding:20px 24px; margin: 16px 0; }}
.no-match-title {{ font-size:24px; font-weight:800; color:#f87171; margin:0; }}

.candidate-row {{ display:flex; align-items:center; gap:12px; padding:7px 0;
    border-bottom:1px solid {PANEL_BORDER}; font-size:13.5px; }}
.candidate-name {{ width:230px; color:{TEXT}; flex-shrink:0; }}
.candidate-bar-track {{ flex:1; background:{PANEL_BORDER}; border-radius:3px; height:8px; }}
.candidate-bar-fill {{ height:8px; border-radius:3px; }}
.candidate-score {{ width:60px; text-align:right; font-family:'JetBrains Mono',monospace;
    color:{TEXT_MUTED}; font-size:12px; }}

div[data-testid="stFileUploader"] {{
    background:{PANEL}; border:1px dashed {PANEL_BORDER}; border-radius:10px;
    padding: 4px;
}}
.stButton button {{
    background:{TEAL}; color:#022922; border:none; border-radius:6px;
    font-weight:600; padding:0.5rem 1.2rem;
}}
.stButton button:hover {{ background:#5eead4; color:#022922; }}

div[data-baseweb="tab-list"] {{ border-bottom: 1px solid {PANEL_BORDER}; gap: 28px; }}
button[data-baseweb="tab"] {{
    font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:0.08em;
    text-transform:uppercase; color:{TEXT_FAINT}; background:transparent;
}}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{TEAL}; }}
div[data-baseweb="tab-highlight"] {{ background-color:{TEAL}; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Matplotlib dark theme to match the demo's plots
# ----------------------------------------------------------------------
def _style_dark_ax(fig, ax):
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor("#070b0d")
    ax.tick_params(colors=TEXT_FAINT, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(PANEL_BORDER)
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    ax.grid(False)


def plot_spectrogram_dark(t, f, Sxx_db):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=130)
    ax.pcolormesh(t, f, Sxx_db, shading="auto", cmap="magma")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    _style_dark_ax(fig, ax)
    fig.tight_layout()
    return fig


def plot_constellation_dark(t, f, peaks, n_peaks_label=True):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=130)
    if len(peaks) > 0:
        pf = [f[p[0]] for p in peaks]
        pt = [t[p[1]] for p in peaks]
        ax.scatter(pt, pf, s=4, c=TEAL, linewidths=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("freq (Hz)")
    if n_peaks_label:
        ax.text(0.98, 0.95, f"{len(peaks)} peaks", transform=ax.transAxes,
                 ha="right", va="top", color=TEXT_FAINT, fontsize=8.5,
                 family="monospace")
    _style_dark_ax(fig, ax)
    fig.tight_layout()
    return fig


def plot_full_fingerprint_dark(times, freqs, window_start, window_end):
    fig, ax = plt.subplots(figsize=(10.5, 3.0), dpi=130)
    ax.scatter(times, freqs, s=3, c=ORANGE, linewidths=0, alpha=0.85)
    ax.axvspan(window_start, window_end, color=TEAL, alpha=0.12)
    ax.axvline(window_start, color=TEAL, linewidth=1)
    ax.axvline(window_end, color=TEAL, linewidth=1)
    ax.set_xlabel("time (frames)")
    ax.set_ylabel("freq bin")
    _style_dark_ax(fig, ax)
    fig.tight_layout()
    return fig


def plot_offset_spike_dark(offset_hist, best_offset, cluster_score):
    fig, ax = plt.subplots(figsize=(10.5, 2.6), dpi=130)
    if len(offset_hist) > 0:
        offsets = sorted(offset_hist.keys())
        counts = [offset_hist[o] for o in offsets]
        ax.bar(offsets, counts, width=2.5, color=ORANGE)
        ax.annotate(f"{cluster_score:,} hashes\nalign here",
                     xy=(best_offset, cluster_score),
                     xytext=(best_offset + (max(offsets)-min(offsets))*0.08 + 5,
                             cluster_score * 0.72),
                     color=ORANGE, fontsize=9, family="monospace",
                     arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    ax.set_xlabel("time offset (database frame - query frame)")
    ax.set_ylabel("# hashes")
    _style_dark_ax(fig, ax)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
@st.cache_resource
def get_database():
    if os.path.exists(DB_PATH):
        return FingerprintDB.load(DB_PATH)
    elif os.path.isdir(SONG_FOLDER):
        return build_database(SONG_FOLDER, db_path=DB_PATH)
    else:
        st.error(f"Could not find `{DB_PATH}` or `{SONG_FOLDER}/`. "
                 "Make sure the song database ships with this app.")
        st.stop()


@st.cache_data(show_spinner=False)
def get_all_song_fingerprints_and_counts(_db_id):
    """
    One-time pass over the whole hash table: for every song, collect its
    full (times, freqs) fingerprint and its total hash count. Cached by
    Streamlit so the Library tab only pays this cost once per session
    instead of once per song per render.
    """
    db = get_database()
    times_by_song = {sid: [] for sid in db.song_names}
    freqs_by_song = {sid: [] for sid in db.song_names}
    counts = {sid: 0 for sid in db.song_names}
    for (f1, f2, dt), entries in db.hash_table.items():
        for sid, t in entries:
            times_by_song[sid].append(t)
            freqs_by_song[sid].append(f1)
            counts[sid] += 1
    fingerprints = {
        sid: (np.array(times_by_song[sid]), np.array(freqs_by_song[sid]))
        for sid in db.song_names
    }
    return fingerprints, counts


def render_song_card_grid(db):
    """Render every song in the library as a small constellation-thumbnail
    card with its hash count, in a 4-column grid."""
    fingerprints, counts = get_all_song_fingerprints_and_counts(id(db))

    names_sorted = sorted(db.song_names.items(), key=lambda kv: kv[1])
    cols = st.columns(4)
    for i, (song_id, name) in enumerate(names_sorted):
        col = cols[i % 4]
        with col:
            times, freqs = fingerprints[song_id]
            color = song_accent_color(name)
            fig, ax = plt.subplots(figsize=(2.4, 1.5), dpi=100)
            fig.patch.set_facecolor("#0a1416")
            ax.set_facecolor("#0a1416")
            if len(times) > 0:
                ax.scatter(times, freqs, s=1.2, c=color, linewidths=0, alpha=0.85)
            ax.axis("off")
            fig.tight_layout(pad=0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown(f"""
                <div style="margin-top:-14px; padding-bottom:14px;">
                  <div class="song-name">{name}</div>
                  <div class="song-hashes">{counts[song_id]:,} hashes</div>
                </div>
            """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(f"""
<div class="app-header">
  <div class="app-icon">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
      <rect x="3" y="9" width="2.5" height="6" rx="1" fill="{TEAL}"/>
      <rect x="8" y="5" width="2.5" height="14" rx="1" fill="{TEAL}"/>
      <rect x="13" y="2" width="2.5" height="20" rx="1" fill="{TEAL}"/>
      <rect x="18" y="7" width="2.5" height="10" rx="1" fill="{TEAL}"/>
    </svg>
  </div>
  <div>
    <p class="app-title">Zapptain America</p>
    <p class="app-sub">Signals, Systems &amp; Networks &middot; Project Demo</p>
  </div>
</div>
<p class="app-tagline">Index a library of songs as spectrogram fingerprints, then identify any short clip against it.</p>
""", unsafe_allow_html=True)

db = get_database()

tab_library, tab_identify, tab_batch = st.tabs(["Library", "Identify", "Batch"])

# ----------------------------------------------------------------------
# LIBRARY TAB
# ----------------------------------------------------------------------
with tab_library:
    st.markdown('<div class="eyebrow">library</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card" style="margin-bottom:24px;">
      <div style="text-align:center; color:{TEXT_MUTED}; font-size:13.5px; line-height:1.6;">
        Song indexing is managed by the admin.<br>
        Drop a clip in the <b style="color:{TEXT};">Identify</b> tab to test the library.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="eyebrow">in the database &middot; {len(db.song_names)} songs</div>',
                unsafe_allow_html=True)
    render_song_card_grid(db)

# ----------------------------------------------------------------------
# IDENTIFY TAB
# ----------------------------------------------------------------------
with tab_identify:
    st.markdown('<div class="eyebrow">search</div>', unsafe_allow_html=True)
    st.markdown("### Identify a clip")

    uploaded = st.file_uploader("Upload a query clip", type=["wav", "mp3", "flac", "ogg", "m4a"],
                                 label_visibility="collapsed")

    query_bytes = None
    if uploaded is not None:
        query_bytes = uploaded.read()

    # Sample clips (if a samples/ folder is shipped with the app)
    if os.path.isdir(SAMPLES_FOLDER):
        sample_files = sorted([f for f in os.listdir(SAMPLES_FOLDER)
                                if f.lower().endswith((".wav", ".mp3", ".flac"))])
        if sample_files:
            st.markdown('<div class="eyebrow" style="margin-top:18px;">or try a sample</div>',
                        unsafe_allow_html=True)
            for sf in sample_files:
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.audio(os.path.join(SAMPLES_FOLDER, sf))
                with c2:
                    if st.button("Try", key=f"try_{sf}"):
                        with open(os.path.join(SAMPLES_FOLDER, sf), "rb") as fh:
                            query_bytes = fh.read()

    run = st.button("Identify", type="primary") if query_bytes else False

    if query_bytes and (run or uploaded is not None):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
            tmp.write(query_bytes)
            tmp_path = tmp.name

        with st.spinner("Fingerprinting and matching..."):
            x, fs = load_audio(tmp_path)
            res = match_query_detailed(x, fs, db, nperseg=4096, neighborhood=(20, 20),
                                        amp_min_db=-60,
                                        confidence_threshold=CONFIDENCE_THRESHOLD)
        os.unlink(tmp_path)

        # --- timing strip ---
        t = res["timings"]
        stages = [
            ("1", "spectrogram", t["spectrogram"], f"{res['Sxx_db'].shape[0]}x{res['Sxx_db'].shape[1]}"),
            ("2", "constellation", t["constellation"], f"{len(res['peaks'])} peaks"),
            ("3", "hashing", t["hashing"], f"{len(res['hashes']):,} hashes"),
            ("4", "db lookup", t["db_lookup"], f"{len(db.song_names)} tracks"),
            ("5", "scoring", t["scoring"], f"offset {res['best_offset']}" if res["best_offset"] is not None else "-"),
        ]
        chips_html = "".join([
            f"""<div class="timing-chip">
                  <div class="timing-num">{n}</div>
                  <div class="timing-label">{label}</div>
                  <div class="timing-ms">{ms:.0f} ms</div>
                  <div class="timing-sub">{sub}</div>
                </div>""" for n, label, ms, sub in stages
        ])
        st.markdown(f"""
        <div class="timing-strip">
          {chips_html}
          <div class="timing-total">total {t['total']:.0f} ms</div>
        </div>
        """, unsafe_allow_html=True)

        # --- match card ---
        if res["best_name"] is None:
            st.markdown(f"""
            <div class="no-match-card">
              <div class="match-eyebrow" style="color:#f87171;">no match</div>
              <p class="no-match-title">No candidate cleared the confidence threshold</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            margin_txt = f"{res['margin_x']:.0f}x the runner-up" if res["margin_x"] else "no runner-up"
            st.markdown(f"""
            <div class="match-card">
              <div class="match-eyebrow">match found</div>
              <p class="match-title">{res['best_name']}</p>
              <p class="match-score">cluster score <b>{res['cluster_score']:,}</b> &middot; <b>{margin_txt}</b></p>
            </div>
            """, unsafe_allow_html=True)

        # --- candidate scores ---
        if res["candidates"]:
            st.markdown('<div class="eyebrow" style="margin-top:22px;">candidate scores</div>',
                        unsafe_allow_html=True)
            max_score = max(s for _, s in res["candidates"]) or 1
            rows_html = ""
            for name, score in res["candidates"]:
                pct = max(2, int(100 * score / max_score))
                color = TEAL if name == res["best_name"] else TEXT_FAINT
                rows_html += f"""
                <div class="candidate-row">
                  <div class="candidate-name">{name}</div>
                  <div class="candidate-bar-track">
                    <div class="candidate-bar-fill" style="width:{pct}%; background:{color};"></div>
                  </div>
                  <div class="candidate-score">{score:,}</div>
                </div>"""
            st.markdown(f'<div class="card">{rows_html}</div>', unsafe_allow_html=True)

        if res["best_name"] is not None:
            # --- Step 1: feature extraction ---
            st.markdown(f"""
            <div class="step-block">
              <div class="step-eyebrow">step 1 &middot; feature extraction</div>
              <p class="step-title">From spectrogram to constellation</p>
              <p class="step-body">The clip was converted into a time-frequency map (left); brighter
              means louder at that frequency and moment. From that rich image, only the
              <span class="hl-teal">{len(res['peaks'])} most prominent peaks</span> were kept (right).
              Discarding amplitude and phase makes the fingerprint robust to EQ, volume changes, and mild noise.</p>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_spectrogram_dark(res["t"], res["f"], res["Sxx_db"]),
                          use_container_width=True)
            with c2:
                st.pyplot(plot_constellation_dark(res["t"], res["f"], res["peaks"]),
                          use_container_width=True)

            # --- Step 2: database search ---
            fingerprints, _ = get_all_song_fingerprints_and_counts(id(db))
            full_times, full_freqs = fingerprints[res["best_song_id"]]
            window_start = res["best_offset"] if res["best_offset"] else 0
            window_end = window_start + len(res["t"])
            st.markdown(f"""
            <div class="step-block">
              <div class="step-eyebrow">step 2 &middot; database search</div>
              <p class="step-title">Where in the song?</p>
              <p class="step-body">The <span class="hl-teal">{len(res['hashes']):,} fingerprint hashes</span>
              were looked up against every indexed track. Below is the full fingerprint of
              <b>{res['best_name']}</b> reconstructed from the database, each dot a stored hash anchor.
              The highlighted window is exactly where the query clip sits inside the full song.</p>
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(plot_full_fingerprint_dark(full_times, full_freqs, window_start, window_end),
                      use_container_width=True)

            # --- Step 3: the proof ---
            st.markdown(f"""
            <div class="step-block">
              <div class="step-eyebrow">step 3 &middot; the proof</div>
              <p class="step-title">The alignment spike</p>
              <p class="step-body">Every matched hash votes for a time offset (database frame minus
              query frame). Chance matches scatter votes randomly, forming a flat noise floor.
              A genuine match makes them converge:
              <span class="hl-orange">{res['cluster_score']:,} hashes agreed on a single offset</span>.
              That spike cannot be a coincidence.</p>
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(plot_offset_spike_dark(res["offset_hist"], res["best_offset"], res["cluster_score"]),
                      use_container_width=True)

# ----------------------------------------------------------------------
# BATCH TAB
# ----------------------------------------------------------------------
with tab_batch:
    st.markdown('<div class="eyebrow">batch</div>', unsafe_allow_html=True)
    st.markdown("### Identify many clips at once")
    st.markdown(f"""
    <p style="color:{TEXT_MUTED}; font-size:13.5px; line-height:1.6; max-width:760px;">
    Upload a set of query clips. Each is identified against the currently indexed library,
    and the results are written to a standardised <code>results.csv</code> with columns
    <code>filename, prediction</code>. The <code>prediction</code> is the matched track's
    filename without its extension, or <code>none</code> when no candidate clears the
    confidence threshold.
    </p>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader("Upload query clips", type=["wav", "mp3", "flac", "ogg", "m4a"],
                                    accept_multiple_files=True, label_visibility="collapsed")

    if batch_files and st.button("Run batch", type="primary"):
        rows = []
        progress_text = st.empty()
        progress_bar = st.progress(0)
        for i, uploaded in enumerate(batch_files):
            progress_text.markdown(f"Identifying ... {i+1}/{len(batch_files)}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            x, fs = load_audio(tmp_path)
            res = match_query_detailed(x, fs, db, nperseg=4096, neighborhood=(20, 20),
                                        amp_min_db=-60, confidence_threshold=CONFIDENCE_THRESHOLD)
            prediction = res["best_name"] if (res["best_name"] and res["is_confident"]) else "none"
            rows.append({"filename": uploaded.name, "prediction": prediction})
            os.unlink(tmp_path)
            progress_bar.progress((i + 1) / len(batch_files))

        n_matched = sum(1 for r in rows if r["prediction"] != "none")
        st.markdown('<div class="eyebrow" style="margin-top:18px;">results</div>',
                    unsafe_allow_html=True)
        st.table(rows)
        st.markdown(f"""<p class="mono" style="color:{TEXT_MUTED}; font-size:12.5px;">
        {n_matched} / {len(rows)} clips matched to a track
        ({len(rows)-n_matched} returned <code>none</code>).</p>""", unsafe_allow_html=True)

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["filename", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("Download results.csv", buf.getvalue(),
                            file_name="results.csv", mime="text/csv")
