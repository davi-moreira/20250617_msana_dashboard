# app.py  – Streamlit ≥ 1.45
# to run: streamlit run app.py --server.maxUploadSize 1024
# ---------------------------------------------------------------------------
# Dashboard for SNA signal analysis
#   • RMS Int SNA (τ = 0.7 s)                     → “RMS Int SNA” tab
#   • Polynomial baseline (deg = 3) on Int SNA    → “Poly Baseline” tab
#   • Mean of corrected Int SNA                   → “Corrected Avg” tab
#   • Artifact detection (≥ μ + 0.3σ)             → “Artifacts” tab
#   • BP-variability analysis (3 mmHg bins)       → “BP Variability Bins” tab
#   • Every interactive plot capped at PLOT_CAP points
# ---------------------------------------------------------------------------
from io import BytesIO, TextIOWrapper
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.signal import find_peaks, butter, filtfilt
import streamlit as st
from numpy.polynomial.polynomial import Polynomial
from sklearn.linear_model import LinearRegression   

# ───────── CONFIGURATION & CONSTANTS ─────────
st.set_page_config(page_title="Signal Dashboard", layout="wide")

COL_NAMES = ["time", "ECG", "Raw SNA", "Ultrasound - Diameter",
             "Finger Pressure", "Heart Rate", "Int SNA"]
USECOLS   = [0, 3, 4, 5, 8, 12, 13]
DTYPES    = {i: "float32" for i in USECOLS}
CHUNK     = 500_000
PLOT_CAP  = 100_000          # max points per interactive trace
TAU       = 0.7              # s for RMS window
POLY_DEG  = 3                # polynomial degree
ARTF_K    = 0.3              # k·σ above mean defines artifact

# --- noise-removal parameters ----------------------------------------------
ASCENT_TH   = 1.5e-7
DESCENT_TH  = 1.5e-7
ASCENT_T    = 0.0004
DESCENT_TW  = 0.5
WAIT_T      = 0.2
SRATE       = 10_000

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def safe_weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """
    Weighted Pearson R that returns np.nan (not a warning) when either
    variable is constant.  If w is None an un-weighted R is returned.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if w is None:
        w = np.ones_like(x)
    m_x, m_y = np.average(x, weights=w), np.average(y, weights=w)
    v_x = np.average((x - m_x) ** 2, weights=w)
    v_y = np.average((y - m_y) ** 2, weights=w)
    if v_x == 0 or v_y == 0:
        return np.nan            # << avoids divide-by-zero
    cov = np.average((x - m_x) * (y - m_y), weights=w)
    return cov / np.sqrt(v_x * v_y)

def identify_and_remove_noise(x, interpolate=False):
    """Return (list_of_index_pairs, cleaned_array)."""
    x = np.asarray(x)
    n            = len(x)
    ascent_pts   = int(ASCENT_T  * SRATE)
    wait_pts     = int(WAIT_T    * SRATE)
    descend_win  = int(DESCENT_TW * SRATE)
    noise_spans  = []
    clean        = x.copy()

    i = 0
    while i < n - ascent_pts:
        if (x[i + ascent_pts] - x[i]) > ASCENT_TH:
            start = i
            end_ascent = i + ascent_pts
            for j in range(end_ascent, min(end_ascent + descend_win, n)):
                if (x[end_ascent] - x[j]) > DESCENT_TH:
                    noise_spans.append((start, j))
                    i = end_ascent + wait_pts
                    break
            else:
                i += wait_pts
        else:
            i += 1

    for a, b in noise_spans:
        clean[a:b + 1] = np.nan

    if interpolate:
        idx = np.arange(n)
        nan = np.isnan(clean)
        clean[nan] = np.interp(idx[nan], idx[~nan], clean[~nan])

    return noise_spans, clean


# ───────── DATA LOADER (cached) ─────────
@st.cache_data(show_spinner="Parsing & preprocessing…")
def load_and_prepare(uploaded_bytes: bytes) -> pd.DataFrame:
    """Load, truncate to 600 s, compute RMS Int SNA."""
    fh = TextIOWrapper(uploaded_bytes, encoding="utf-8", errors="replace")
    chunks, t0 = [], None

    for chunk in pd.read_csv(
        fh, sep=r"\s+", header=None, usecols=USECOLS,
        dtype=DTYPES, on_bad_lines="skip", engine="python", chunksize=CHUNK
    ):
        if t0 is None:
            t0 = float(chunk.iloc[0, 0])
        chunk.iloc[:, 0] -= t0
        chunk = chunk[chunk.iloc[:, 0] <= 600]
        if chunk.empty:
            break
        chunks.append(chunk)
        if chunk.iloc[-1, 0] >= 600:
            break

    df = pd.concat(chunks, ignore_index=True)
    df.columns = COL_NAMES

    # RMS Int SNA --------------------------------------------------
    df["Rectified Raw SNA"] = df["Raw SNA"].abs()
    samp = float(df["time"].iloc[1] - df["time"].iloc[0])
    win  = max(1, int(TAU / samp))
    df["RMS Int SNA"] = (
        df["Rectified Raw SNA"]
        .rolling(window=win, min_periods=1)
        .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
    )
    df.loc[: win - 1, "RMS Int SNA"] = np.nan
    return df


# ───────── SIDEBAR ─────────
logo_path = Path(__file__).parent / "figs" / "lab_logo.png"
if logo_path.is_file():
    st.sidebar.image(str(logo_path), width=210)

st.sidebar.markdown("### 1  Select data")
up_file = st.sidebar.file_uploader("📁 Whitespace .txt", type="txt")
if st.sidebar.button("▶ Load data", disabled=up_file is None):
    st.session_state.df = load_and_prepare(up_file)

df: pd.DataFrame | None = st.session_state.get("df")

if df is not None:
    st.sidebar.caption(f"Loaded **{len(df):,} rows × {df.shape[1]} cols** (≤ 600 s).")
    st.sidebar.markdown("### 2  Choose variables")
    default_cols = COL_NAMES[1:4]
    cols = st.sidebar.multiselect("Variables for analysis:", COL_NAMES,
                                  default=default_cols, key="var_select")
else:
    cols = []

if df is None:
    st.info("Upload a file and press **Load data**.")
    st.stop()

# ───────── PRE-COMPUTE POLYNOMIAL CORRECTION ─────────
poly_df = df[["time", "Int SNA"]].dropna()
p_fit   = Polynomial.fit(poly_df["time"], poly_df["Int SNA"], POLY_DEG)
baseline = p_fit(poly_df["time"])
corr_int = poly_df["Int SNA"] - baseline
disp_df  = poly_df.assign(Baseline=baseline, Corrected=corr_int).reset_index()

avg_val  = corr_int.mean()
std_val  = corr_int.std()
thr_val  = avg_val + ARTF_K * std_val
disp_df["Artifact"] = np.where(disp_df["Corrected"] >= thr_val,
                               disp_df["Corrected"], np.nan)

spans, clean_arr = identify_and_remove_noise(disp_df["Artifact"], interpolate=True)
disp_df["Artifact Clean"] = clean_arr

# ───────── TABS ─────────

tab_over, tab_time, tab_corr, tab_rms, tab_poly, tab_avg, \
tab_art, tab_clean, tab_peaks, tab_nopeaks, tab_bladj, \
tab_filt, tab_msna, tab_bp, tab_baro, tab_cd, tab_neural, \
tab_neural_cmp, tab_mecharc, tab_mecharc_sel = st.tabs(
    ["Overview", "Time series", "Correlation",
     "RMS Int SNA", "Poly Baseline", "Corrected Avg",
     "Artifacts", "Cleaned Artifacts", "Burst Detection",
     "No-Peaks", "BL-Adjusted", "Filtered Peaks",
     "MSNA Quantification", "BP Variability Bins",
     "Baroreflex Slope",
     "Carotid Diameter Troughs",
     "Neural Arc (C diameter)", "Neural Arc – Orig vs Sel",
     "Mechanical Arc",                  
     "Mechanical Arc (Sel)"]            
)


# 1 Overview ------------------------------------------------------
with tab_over:
    st.subheader("Dataset preview")
    st.dataframe(df.head())
    if cols:
        st.write("Summary statistics", df[cols].describe())

# 2 Time-series ---------------------------------------------------
with tab_time:
    st.markdown(f"<small>Each trace subsampled to <b>{PLOT_CAP:,}</b> points.</small>",
                unsafe_allow_html=True)
    if cols:
        for c in cols:
            ts = df[["time", c]].reset_index().rename(columns={"index": "obs_idx"})
            if len(ts) > PLOT_CAP:
                ts = ts.sample(PLOT_CAP).sort_values("time")
            st.plotly_chart(px.line(ts, x="time", y=c, title=c,
                                     hover_data=["obs_idx"]),
                            use_container_width=True)
    else:
        st.info("Select at least one variable.")

# 3 Correlation ---------------------------------------------------
with tab_corr:
    if len(cols) > 1:
        corr_df = df[cols]
        if len(corr_df) > PLOT_CAP:
            corr_df = corr_df.sample(PLOT_CAP)
        st.plotly_chart(px.scatter_matrix(corr_df, dimensions=cols),
                        use_container_width=True)
    else:
        st.info("Select two or more variables.")

# 4 RMS Int SNA ---------------------------------------------------
with tab_rms:
    st.markdown(
        f"<small>RMS window = {TAU:.1f}s; subsampled to <b>{PLOT_CAP:,}</b>.</small>",
        unsafe_allow_html=True)
    rms_df = df[["time", "RMS Int SNA"]].dropna().reset_index()
    if len(rms_df) > PLOT_CAP:
        rms_df = rms_df.sample(PLOT_CAP).sort_values("time")
    st.plotly_chart(px.line(rms_df, x="time", y="RMS Int SNA",
                            title="Interactive RMS Integrated Sympathetic Nerve Activity"), 
                            use_container_width=True)
    st.subheader("First & last 4 rows")
    st.dataframe(pd.concat([df.head(4), df.tail(4)])
                 [["time", "Raw SNA", "Rectified Raw SNA", "RMS Int SNA"]],
                 height=240)
# 5 Poly Baseline -------------------------------------------------
with tab_poly:
    st.markdown(f"<small>Polynomial degree = {POLY_DEG}; cap = {PLOT_CAP:,}.</small>",
                unsafe_allow_html=True)

    plot_df = disp_df
    if len(plot_df) > PLOT_CAP:
        plot_df = plot_df.sample(PLOT_CAP).sort_values("time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["Int SNA"],
                             mode="lines", name="Original"))
    fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["Baseline"],
                             mode="lines", name="Baseline"))
    fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["Corrected"],
                             mode="lines", name="Corrected"))
    fig.update_layout(title="Corrected Integrated SNA (Polynomial Fitting)",
                      xaxis_title="Time (s)", yaxis_title="Integrated SNA")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("First & last 4 rows")
    st.dataframe(pd.concat([disp_df.head(4), disp_df.tail(4)])
                 [["time", "Int SNA", "Baseline", "Corrected"]],
                 height=240)

# 6 Corrected Avg -------------------------------------------------
with tab_avg:
    st.markdown(
        f"<small>Dashed red line = global mean. Cap = {PLOT_CAP:,} points.</small>",
        unsafe_allow_html=True)
    st.metric("Average Corrected Int SNA", f"{avg_val:.3f}")
    avg_df = disp_df.rename(columns={"Corrected": "Corrected Int SNA (Poly)"})
    plot_df = avg_df[["time", "Corrected Int SNA (Poly)"]]
    if len(plot_df) > PLOT_CAP:
        plot_df = plot_df.sample(PLOT_CAP).sort_values("time")
    fig_avg = go.Figure()
    fig_avg.add_trace(go.Scatter(x=plot_df["time"],
                                 y=plot_df["Corrected Int SNA (Poly)"],
                                 mode="lines", name="Corrected Int SNA (Poly)"))
    fig_avg.add_trace(go.Scatter(x=[plot_df["time"].min(), plot_df["time"].max()],
                                 y=[avg_val, avg_val], mode="lines",
                                 name=f"Average ({avg_val:.2f})",
                                 line=dict(color="red", dash="dash")))
    fig_avg.update_layout(title="Interactive Integrated Sympathetic Nerve Activity with Average",
                          xaxis_title="Time (s)", yaxis_title="Integrated SNA",
                          template="plotly_white")
    st.plotly_chart(fig_avg, use_container_width=True)
    st.subheader("First & last 4 rows")
    st.dataframe(pd.concat([avg_df.head(4), avg_df.tail(4)])
                 [["time","Corrected Int SNA (Poly)"]], height=240)

# 7 Artifacts (new) -----------------------------------------------
with tab_art:
    st.markdown(
        (f"<small>Artifacts = points ≥ μ + {ARTF_K}·σ "
         f"(μ ≈ {avg_val:.3f}, σ ≈ {std_val:.3f}, "
         f"threshold ≈ {thr_val:.3f}). Cap = {PLOT_CAP:,}.</small>"),
        unsafe_allow_html=True)
    art_df = disp_df[["time", "Artifact"]].dropna()
    if len(art_df) > PLOT_CAP:
        art_df = art_df.sample(PLOT_CAP).sort_values("time")
    fig_art = go.Figure()
    fig_art.add_trace(go.Scatter(x=art_df["time"], y=art_df["Artifact"],
                                 mode="lines", name="Artifacts"))
    fig_art.add_trace(go.Scatter(x=[art_df["time"].min(), art_df["time"].max()],
                                 y=[thr_val, thr_val], mode="lines",
                                 name=f"Threshold ({thr_val:.2f})",
                                 line=dict(color="green", dash="dash")))
    fig_art.add_trace(go.Scatter(x=[art_df["time"].min(), art_df["time"].max()],
                                 y=[avg_val, avg_val], mode="lines",
                                 name=f"Mean ({avg_val:.2f})",
                                 line=dict(color="red", dash="dash")))
    fig_art.update_layout(title="Interactive Integrated Sympathetic Nerve Activity with Artifacts",
                          xaxis_title="Time (s)", yaxis_title="Integrated SNA",
                          template="plotly_white")
    st.plotly_chart(fig_art, use_container_width=True)

    st.subheader("First & last 4 rows")
    st.dataframe(pd.concat([disp_df.head(4), disp_df.tail(4)])
                 [["time","Corrected","Artifact"]], height=240)
    
# 8 Cleaned Artifacts  (auto + manual) ----------------------------------------
with tab_clean:
    st.markdown(
        f"<small>Auto-detected noise spans: <b>{len(spans)}</b>. "
        "Use the controls below to add / clear extra spans.<br>"
        "Grey = raw Artifact • Sky-blue = auto-cleaned • Orange = manual.</small>",
        unsafe_allow_html=True,
    )

    # ── manual-edit UI ───────────────────────────────────────────────
    if "manual_spans" not in st.session_state:
        st.session_state.manual_spans = []

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        t_start = st.number_input("Start time [s]", value=0.0, step=0.1, key="man_start")
    with col2:
        t_end   = st.number_input("End time [s]",   value=0.1, step=0.1, key="man_end")
    with col3:
        if st.button("➕ Add span"):
            if t_end > t_start:
                st.session_state.manual_spans.append((t_start, t_end))

    if st.button("🔄 Reset manual spans"):
        st.session_state.manual_spans.clear()

    # ── build cleaned-plus-manual signal ────────────────────────────
    manual_clean = clean_arr.copy()
    for t0, t1 in st.session_state.manual_spans:
        i0 = np.abs(disp_df["time"] - t0).idxmin()
        i1 = np.abs(disp_df["time"] - t1).idxmin()
        manual_clean[i0:i1 + 1] = np.nan

    # linear interpolation over new NaNs
    idx_all = np.arange(len(manual_clean))
    nan_mask = np.isnan(manual_clean)
    if nan_mask.any():
        manual_clean[nan_mask] = np.interp(idx_all[nan_mask],
                                           idx_all[~nan_mask],
                                           manual_clean[~nan_mask])

    disp_df["Artifact Manual"] = manual_clean
    avg_clean = float(np.nanmean(manual_clean))          # ← NEW mean of cleaned

    # quick KPI
    st.metric("Mean (cleaned)", f"{avg_clean:.3e}")

    # ── Plot ────────────────────────────────────────────────────────
    plot_df = disp_df[["time", "Artifact",
                       "Artifact Clean", "Artifact Manual"]]
    if len(plot_df) > PLOT_CAP:
        plot_df = plot_df.sample(PLOT_CAP).sort_values("time")

    fig_cl = go.Figure()
    fig_cl.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["Artifact"],
                                name="Original", line=dict(color="darkgray")))
    fig_cl.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["Artifact Clean"],
                                name="Auto-clean", line=dict(color="skyblue")))
    fig_cl.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["Artifact Manual"],
                                name="Auto + Manual", line=dict(color="orange")))
    # dashed mean line
    fig_cl.add_trace(go.Scatter(
        x=[plot_df["time"].min(), plot_df["time"].max()],
        y=[avg_clean, avg_clean],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name=f"Mean ({avg_clean:.2e})",
    ))
    # shade all spans
    for a, b in spans + [
        (np.abs(disp_df["time"]-s).idxmin(), np.abs(disp_df["time"]-e).idxmin())
        for s, e in st.session_state.manual_spans
    ]:
        fig_cl.add_vrect(x0=disp_df.at[a, "time"], x1=disp_df.at[b, "time"],
                         fillcolor="red", opacity=0.25, line_width=0)

    fig_cl.update_layout(title="Artifact Signal – Auto + Manual Cleaning",
                         xaxis_title="Time (s)", yaxis_title="Integrated SNA",
                         template="plotly_white")
    st.plotly_chart(fig_cl, use_container_width=True)

    # ── diagnostics ────────────────────────────────────────────────
    st.subheader("First & last 4 rows")
    st.dataframe(
        pd.concat([disp_df.head(4), disp_df.tail(4)])
          [["time", "Artifact", "Artifact Clean", "Artifact Manual"]],
        height=240,
    )

# 9 Burst Detection -----------------------------------------------------------
with tab_peaks:
    # ── parameters -----------------------------------------------------------
    time_start, time_end = 0, 300          # x-axis window
    prom      = 2e-7                       # peak prominence
    samp_int  = float(df["time"].iloc[1] - df["time"].iloc[0])
    dist_pts  = int(0.4 / samp_int)        # 0.4 s as samples

    # cleaned signal from previous step
    y_clean   = manual_clean
    avg_c     = float(np.nanmean(y_clean))
    std_c     = float(np.nanstd(y_clean))
    thr_c     = avg_c + 0.3 * std_c

    # detect peaks (ignore NaNs)
    valid     = ~np.isnan(y_clean)
    peaks, _  = find_peaks(y_clean[valid], height=thr_c,
                           distance=dist_pts, prominence=prom)
    # convert indices back to original axis
    peaks_idx = np.where(valid)[0][peaks]

    # KPI read-outs
    st.metric("Mean (cleaned)",      f"{avg_c:.3e}")
    st.metric("Std Dev (cleaned)",   f"{std_c:.3e}")
    st.metric("Threshold (cleaned)", f"{thr_c:.3e}")
    st.metric("Detected bursts",     f"{len(peaks_idx)}")

    # ── plot ---------------------------------------------------------------
    plot_df = pd.DataFrame({
        "time": disp_df["time"],
        "Clean": y_clean
    })
    if len(plot_df) > PLOT_CAP:
        plot_df = plot_df.sample(PLOT_CAP).sort_values("time")

    fig_pk = go.Figure()
    fig_pk.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["Clean"],
                                mode="lines", name="Cleaned Int SNA",
                                line=dict(color="darkgray")))
    fig_pk.add_trace(go.Scatter(
        x=[time_start, time_end], y=[avg_c, avg_c],
        mode="lines", name="Mean", line=dict(color="red", dash="dash")
    ))
    fig_pk.add_trace(go.Scatter(
        x=[time_start, time_end], y=[thr_c, thr_c],
        mode="lines", name="Threshold", line=dict(color="green", dash="dash")
    ))
    fig_pk.add_trace(go.Scatter(
        x=disp_df["time"].iloc[peaks_idx],
        y=y_clean[peaks_idx],
        mode="markers", name="Peaks", marker=dict(color="red", size=6)
    ))

    fig_pk.update_layout(
        title="Cleaned Int SNA with Burst Detection",
        xaxis_title="Time (s)", yaxis_title="Integrated SNA",
        xaxis_range=[time_start, time_end],
        yaxis_range=[0, 4e-6],
        template="plotly_white"
    )
    st.plotly_chart(fig_pk, use_container_width=True)

# 10  No-Peaks (cleaned trace with 0.4 s NaN gaps around each burst) ----------
with tab_nopeaks:
    # ---- build Int SNA_no_peaks ------------------------------------------------
    sna_no_peak = manual_clean.copy()                 # start from cleaned signal
    t_step      = samp_int                            # already computed above
    win_pts     = int(0.4 / t_step)                   # 0.4 s window size

    for pk in peaks_idx:                              # peaks from tab_peaks
        i0 = max(pk - win_pts, 0)
        i1 = min(pk + win_pts, len(sna_no_peak) - 1)
        sna_no_peak[i0:i1 + 1] = np.nan

    # basic stats (ignore NaNs)
    avg_np = float(np.nanmean(sna_no_peak))
    min_np = float(np.nanmin(sna_no_peak))
    max_np = float(np.nanmax(sna_no_peak))

    # KPI strip
    st.metric("Mean (No-Peaks)", f"{avg_np:.3e}")
    st.metric("Min (No-Peaks)",  f"{min_np:.3e}")
    st.metric("Max (No-Peaks)",  f"{max_np:.3e}")

    # ---- plot -----------------------------------------------------------------
    plot_df = pd.DataFrame({"time": disp_df["time"], "NoPk": sna_no_peak})
    if len(plot_df) > PLOT_CAP:
        plot_df = plot_df.sample(PLOT_CAP).sort_values("time")

    fig_np = go.Figure()
    fig_np.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["NoPk"],
                                mode="lines", name="Int SNA_no_peaks",
                                line=dict(color="darkgray")))
    fig_np.add_trace(go.Scatter(
        x=[time_start, time_end], y=[avg_np, avg_np],
        mode="lines", name="Mean", line=dict(color="red", dash="dash")
    ))
    fig_np.add_trace(go.Scatter(
        x=[time_start, time_end], y=[min_np, min_np],
        mode="lines", name="Min",  line=dict(color="blue", dash="dash")
    ))
    fig_np.add_trace(go.Scatter(
        x=[time_start, time_end], y=[max_np, max_np],
        mode="lines", name="Max",  line=dict(color="green", dash="dash")
    ))
    # mark peaks for reference
    fig_np.add_trace(go.Scatter(
        x=disp_df["time"].iloc[peaks_idx],
        y=manual_clean[peaks_idx],
        mode="markers", name="Peaks", marker=dict(color="red", size=5)
    ))
    fig_np.update_layout(
        title="Integrated SNA with 0.4 s NaN gaps around Peaks",
        xaxis_title="Time (s)", yaxis_title="Integrated SNA",
        xaxis_range=[time_start, time_end], template="plotly_white"
    )
    st.plotly_chart(fig_np, use_container_width=True)

    # ---- preview table --------------------------------------------------------
    st.subheader("First & last 4 rows")
    st.dataframe(
        pd.concat([plot_df.head(4), plot_df.tail(4)]).reset_index(drop=True),
        height=240,
    )

# 11  BL-Adjusted  (cleaned signal centred to its own mean) -------------------
with tab_bladj:
    # --- baseline-adjust the cleaned signal -----------------------------------
    bl_adj   = manual_clean - avg_np               # avg_np from tab_nopeaks
    std_bl   = float(np.nanstd(bl_adj))
    noise_lv = avg_np + 0.2 * std_bl               # same formula as spec
    peaks_bl = y_clean[peaks_idx] - avg_np         # peaks already detected

    # KPIs
    st.metric("Std-Dev (BL adj.)", f"{std_bl:.3e}")
    st.metric("Noise level",         f"{noise_lv:.3e}")

    # --- plotting -------------------------------------------------------------
    plot_df = pd.DataFrame({"time": disp_df["time"], "BLadj": bl_adj})
    if len(plot_df) > PLOT_CAP:
        plot_df = plot_df.sample(PLOT_CAP).sort_values("time")

    fig_bl = go.Figure()
    fig_bl.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["BLadj"],
                                mode="lines", name="Int SNA_BLadjusted",
                                line=dict(color="darkgray")))
    # baseline (0) and noise level
    fig_bl.add_trace(go.Scatter(x=[time_start, time_end], y=[0, 0],
                                mode="lines", name="Baseline",
                                line=dict(color="black", dash="dash")))
    fig_bl.add_trace(go.Scatter(x=[time_start, time_end], y=[noise_lv, noise_lv],
                                mode="lines", name="Noise level",
                                line=dict(color="red", dash="dash")))
    # peaks
    fig_bl.add_trace(go.Scatter(x=disp_df["time"].iloc[peaks_idx],
                                y=peaks_bl,
                                mode="markers", name="Peaks",
                                marker=dict(color="red", size=5)))
    fig_bl.update_layout(title="Baseline-Adjusted Integrated SNA",
                         xaxis_title="Time (s)",
                         yaxis_title="Int SNA (BL-adjusted)",
                         xaxis_range=[time_start, time_end],
                         template="plotly_white")
    st.plotly_chart(fig_bl, use_container_width=True)

    # --- table preview --------------------------------------------------------
    st.subheader("First & last 4 rows")
    st.dataframe(
        pd.concat([plot_df.head(4), plot_df.tail(4)]).reset_index(drop=True),
        height=240,
    )

# 12  Filtered Peaks  (only bursts above noise level) -------------------------
with tab_filt:
    # ── select peaks strictly above the noise level computed in tab_bladj ──
    mask_filt          = peaks_bl > noise_lv           # peaks_bl & noise_lv from tab_bladj
    filt_idx           = peaks_idx[mask_filt]          # time-indices of filtered peaks
    peaks_filt_bl      = peaks_bl[mask_filt]           # BL-adjusted peak heights

    # KPIs
    st.metric("Noise level",       f"{noise_lv:.3e}")
    st.metric("Filtered bursts",   f"{len(filt_idx)}")

    # ---- plotting -----------------------------------------------------------
    plot_df = pd.DataFrame({"time": disp_df["time"], "BLadj": bl_adj})
    if len(plot_df) > PLOT_CAP:
        plot_df = plot_df.sample(PLOT_CAP).sort_values("time")

    fig_fp = go.Figure()
    fig_fp.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["BLadj"],
                                mode="lines",
                                name="Int SNA_BLadjusted",
                                line=dict(color="darkgray")))
    # baseline & noise level
    fig_fp.add_trace(go.Scatter(x=[time_start, time_end], y=[0, 0],
                                mode="lines", name="Baseline",
                                line=dict(color="black", dash="dash")))
    fig_fp.add_trace(go.Scatter(x=[time_start, time_end], y=[noise_lv, noise_lv],
                                mode="lines", name="Noise level",
                                line=dict(color="red", dash="dash")))
    # filtered peaks
    fig_fp.add_trace(go.Scatter(
        x=disp_df["time"].iloc[filt_idx],
        y=peaks_filt_bl,
        mode="markers", name="Filtered Peaks",
        marker=dict(color="red", size=6)
    ))

    fig_fp.update_layout(
        title="BL-Adjusted Integrated SNA with Filtered Peaks",
        xaxis_title="Time (s)",
        yaxis_title="Int SNA (BL-adjusted)",
        xaxis_range=[time_start, time_end],
        template="plotly_white"
    )
    st.plotly_chart(fig_fp, use_container_width=True)

    # ---- table preview ------------------------------------------------------
    st.subheader("First & last 4 rows")
    st.dataframe(
        pd.concat([plot_df.head(4), plot_df.tail(4)]).reset_index(drop=True),
        height=240,
    )

    st.session_state["filt_idx"] = filt_idx

# 13 MSNA Quantification & Excel export --------------------------------------
with tab_msna:
    st.markdown("## Muscle Sympathetic Nerve Activity (MSNA) metrics")

    # — ECG R-wave detection —
    samp_int = float(df["time"].iloc[1] - df["time"].iloc[0])

    def bp_filter(sig, fs=1/samp_int, lo=0.5, hi=45, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
        return filtfilt(b, a, sig)

    ecg_filt = bp_filter(df["ECG"].values)
    r_peaks_all, _ = find_peaks(ecg_filt, distance=int(0.3/samp_int))
    thr_ecg  = ecg_filt[r_peaks_all].mean() + 3*ecg_filt[r_peaks_all].std()
    peaks_ecg = r_peaks_all[ecg_filt[r_peaks_all] > thr_ecg]

    # — diastolic BP & carotid minima (lists built here; code unchanged) —
    diastolic_pressure_indices, diastolic_pressure_values = [], []
    diastolic_diameter_indices, diastolic_diameter_values = [], []

    fp = df["Finger Pressure"].values
    cd = df["Ultrasound - Diameter"].values
    t  = df["time"].values

    for r_idx in peaks_ecg:
        r_time = t[r_idx]
        mask_bp = (t >= r_time + 0.30) & (t <= r_time + 1.50)
        if mask_bp.any():
            local_idx   = np.argmin(fp[mask_bp])
            global_idx  = np.where(mask_bp)[0][local_idx]
            diastolic_pressure_indices.append(global_idx)
            diastolic_pressure_values.append(fp[global_idx])

            d_time      = t[global_idx]
            mask_cd     = (t >= d_time) & (t <= d_time + 0.50)
            if mask_cd.any():
                local_idx_cd  = np.argmin(cd[mask_cd])
                global_idx_cd = np.where(mask_cd)[0][local_idx_cd]
                diastolic_diameter_indices.append(global_idx_cd)
                diastolic_diameter_values.append(cd[global_idx_cd])

    diastolic_pressure_indices = np.asarray(diastolic_pressure_indices, dtype=int)
    diastolic_pressure_values  = np.asarray(diastolic_pressure_values,  dtype=float)
    diastolic_diameter_indices = np.asarray(diastolic_diameter_indices, dtype=int)
    diastolic_diameter_values  = np.asarray(diastolic_diameter_values, dtype=float)

    # ── MSNA metrics ────────────────────────────────────────────────────────
    total_time_min   = df["time"].iloc[-1] / 60.0
    total_bursts     = len(filt_idx)                       # from tab_filt
    total_cycles     = len(peaks_ecg)
    burst_freq       = total_bursts / total_time_min if total_time_min else 0
    burst_incidence  = (total_bursts / total_cycles)*100 if total_cycles else 0

    if len(peaks_filt_bl):                                # from tab_filt
        largest_peak  = float(peaks_filt_bl.max())
        scaled_heights = peaks_filt_bl / largest_peak * 100
        mbh            = float(scaled_heights.mean())
    else:
        mbh = 0.0
    total_activity = burst_freq * mbh

    msna_res = {
        "Total Bursts": total_bursts,
        "Total Cardiac Cycles": total_cycles,
        "Burst Frequency (bursts/min)": round(burst_freq, 2),
        "Burst Incidence (bursts/100 beats)": round(burst_incidence, 2),
        "Mean Burst Height (% largest)": round(mbh, 2),
        "Total Activity (Freq × MBH)": round(total_activity, 2),
    }

    # ── KPI display ─────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Burst Freq [bursts/min]",  f"{msna_res['Burst Frequency (bursts/min)']}")
    k2.metric("Burst Incidence [%]",      f"{msna_res['Burst Incidence (bursts/100 beats)']}")
    k3.metric("Total Activity ",          f"{msna_res['Total Activity (Freq × MBH)']}")

    st.table(pd.DataFrame([msna_res]))

    # ── MSNA results download  ────────────────────────────────────────────────
    from io import BytesIO

    df_msna = pd.DataFrame([msna_res])          # as before
    bio      = BytesIO()                        # in-memory buffer

    file_name = "MSNA_results"
    mime_type = ""
    try:
        # try xlsxwriter first (preferred)
        with pd.ExcelWriter(bio, engine="xlsxwriter") as xlw:
            # MSNA sheet (already built)
            df_msna.to_excel(xlw, sheet_name="MSNA Data", index=False)

            # BP variability -------------------------------------------------------
            bp_met  = st.session_state.get("bp_metrics")
            bin_cnt = st.session_state.get("bin_counts")
            baro = st.session_state.get("baro_df")
            diam = st.session_state.get("diam_stats")      
            neural = st.session_state.get("neural_arc_df")
            if baro is not None:
                baro.to_excel(xlw, sheet_name="Baroreflex", index=False)
            if bp_met is not None:
                bp_met.to_excel(xlw, sheet_name="BP Variability", index=False)
            if bin_cnt is not None:
                bin_cnt.to_excel(xlw, sheet_name="Trough Bins", index=False)
             # Carotid-diameter sheets  ─────────────────────────────────────    
            if diam:
                diam["orig_stats"].to_excel(xlw, sheet_name="Original Diameter", index=False)
                start_row = len(diam["orig_stats"]) + 2
                diam["orig_bins"].to_excel(xlw, sheet_name="Original Diameter",
                                        index=False, startrow=start_row)

                diam["sel_stats"].to_excel(xlw, sheet_name="Selected Diameter", index=False)
                start_row = len(diam["sel_stats"]) + 2
                diam["sel_bins"].to_excel(xlw, sheet_name="Selected Diameter",
                                        index=False, startrow=start_row)
            if neural is not None:
                neural.to_excel(xlw, sheet_name="Neural Arc", index=False)

            neural_cmp = st.session_state.get("neural_cmp_df")
            if neural_cmp is not None:
                neural_cmp.to_excel(xlw, sheet_name="Neural Arc (Comp)", index=False)
            neural_orig = st.session_state.get("neural_arm_orig")
            neural_sel  = st.session_state.get("neural_arm_sel")
            if neural_orig is not None:
                neural_orig.to_excel(xlw,
                                     sheet_name="Neural Arm Original",
                                     index=False)
            if neural_sel is not None:
                neural_sel.to_excel(xlw,
                                    sheet_name="Neural Arm Selected",
                                    index=False)
            mech_arc = st.session_state.get("mech_arc_df")
            if mech_arc is not None:
                mech_arc.to_excel(xlw, sheet_name="Mechanical Arc", index=False)
            mech_comp_orig = st.session_state.get("mech_arc_df")
            mech_comp_sel  = st.session_state.get("mech_arc_sel_df")
            if mech_comp_orig is not None:
                mech_comp_orig.to_excel(
                    xlw, sheet_name="Mechanical Component Original", index=False)
            if mech_comp_sel is not None:
                mech_comp_sel.to_excel(
                    xlw, sheet_name="Mechanical Component Selected", index=False)

        file_name += ".xlsx"
        mime_type  = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except ModuleNotFoundError:
        # neither xlsxwriter nor openpyxl found → fall back to CSV
        bio = BytesIO()                         # reset buffer
        df_msna.to_csv(bio, index=False)
        file_name += ".csv"
        mime_type  = "text/csv"

    bio.seek(0)                                 # rewind before sending

    st.download_button(
        "⬇️ Download MSNA results",
        data=bio.getvalue(),
        file_name=file_name,
        mime=mime_type,
        help="Will download .xlsx if Excel writers are installed, "
            "otherwise a .csv file."
    )


# ── FOUR-PANEL DIAGNOSTIC FIGURE ───────────────────────────────────────
    VIS_CAP = PLOT_CAP
    vis_idx = np.linspace(0, len(df) - 1, VIS_CAP, dtype=int) \
              if len(df) > VIS_CAP else np.arange(len(df))

    t_vis   = df["time"].values[vis_idx]
    ecg_vis = ecg_filt[vis_idx]
    fp_vis  = fp[vis_idx]
    cd_vis  = cd[vis_idx]
    bl_vis  = bl_adj[vis_idx]

    # ------------------------------------------------------------------
    # helper full-length time arrays for marker traces (no .iloc needed)
    # ------------------------------------------------------------------
    time_full          = df["time"].values
    ecg_peak_times     = time_full[peaks_ecg]
    dia_bp_times       = time_full[diastolic_pressure_indices]
    dia_diam_times     = time_full[diastolic_diameter_indices]
    burst_peak_times   = time_full[filt_idx]

    fig4 = sp.make_subplots(rows=4, cols=1, shared_xaxes=True,
                            vertical_spacing=0.04,
                            subplot_titles=("ECG + R waves",
                                            "Finger Pressure + Diastole",
                                            "Carotid Diameter + Diastole",
                                            "Integrated SNA + Bursts"))

    # 1 ECG – continuous & markers
    fig4.add_scatter(x=t_vis, y=ecg_vis, row=1, col=1,
                     name="Filtered ECG", line=dict(color="black", width=1))
    fig4.add_scatter(x=ecg_peak_times, y=ecg_filt[peaks_ecg],
                     mode="markers+text", marker=dict(color="red", size=4),
                     text=[str(i+1) for i in range(len(peaks_ecg))],
                     textposition="top center", textfont_size=8,
                     showlegend=False, row=1, col=1)

    # 2 Finger pressure
    fig4.add_scatter(x=t_vis, y=fp_vis, row=2, col=1,
                     name="Finger Pressure", line=dict(color="blue", width=1))
    fig4.add_scatter(x=dia_bp_times, y=diastolic_pressure_values,
                     mode="markers+text", marker=dict(color="red", size=4),
                     text=[str(i+1) for i in range(len(diastolic_pressure_indices))],
                     textposition="bottom center", textfont_size=8,
                     showlegend=False, row=2, col=1)

    # 3 Carotid diameter
    fig4.add_scatter(x=t_vis, y=cd_vis, row=3, col=1,
                     name="Carotid Diameter", line=dict(color="purple", width=1))
    fig4.add_scatter(x=dia_diam_times, y=diastolic_diameter_values,
                     mode="markers+text", marker=dict(color="red", size=4),
                     text=[str(i+1) for i in range(len(diastolic_diameter_indices))],
                     textposition="bottom center", textfont_size=8,
                     showlegend=False, row=3, col=1)

    # 4 MSNA (baseline-adjusted) + burst markers
    fig4.add_scatter(x=t_vis, y=bl_vis, row=4, col=1,
                     name="Int SNA (BL-adj)", line=dict(color="darkgray", width=1))
    fig4.add_scatter(x=burst_peak_times, y=peaks_filt_bl,
                     mode="markers+text", marker=dict(color="green", size=5),
                     text=[str(i+1) for i in range(len(filt_idx))],
                     textposition="top center", textfont_size=8,
                     showlegend=False, row=4, col=1)

    fig4.update_layout(height=900, template="plotly_white",
                       showlegend=False, margin=dict(r=110))
    st.plotly_chart(fig4, use_container_width=True)

    st.session_state["peaks_ecg"] = peaks_ecg

# 14 ───────────────────────────  BP variability tab  ─────────────────────────
with tab_bp:
    st.markdown("## Blood-pressure variability (3 mmHg bins)")

    # — detect troughs & systolic peaks —
    fp_signal = df["Finger Pressure"].values
    time_arr  = df["time"].values
    dt        = float(df["time"].iloc[1] - df["time"].iloc[0])

    prom_tr, thr_tr = 1.5, fp_signal.mean() - fp_signal.std()
    min_dist        = int(0.7 / dt)
    troughs_idx, _  = find_peaks(-fp_signal, prominence=prom_tr,
                                 height=-thr_tr, distance=min_dist)

    prom_sys = 10
    syst_idx, _ = find_peaks(fp_signal, prominence=prom_sys, distance=min_dist)

    syst_vals = fp_signal[syst_idx]
    dia_vals  = fp_signal[troughs_idx]

    # — 3 mmHg bins of diastolic troughs —
    bin_edges = np.arange(fp_signal.min(), fp_signal.max() + 3, 3)
    bin_lbls  = [f"{round(b,1)}–{round(b+3,1)}" for b in bin_edges[:-1]]
    dia_bins  = pd.cut(dia_vals, bins=bin_edges, labels=bin_lbls, include_lowest=True)
    bin_cnts  = dia_bins.value_counts().sort_index()
    valid_cnt = bin_cnts[bin_cnts >= 3]

    st.metric("Detected heart-beats", f"{len(syst_idx)}")
    st.metric("Valid 3 mmHg bins",    f"{len(valid_cnt)}")

    # 1 – Valid-bin counts table
    st.subheader("Valid bin counts (≥ 3 troughs)")
    valid_cnt_df = valid_cnt.reset_index().rename(
        columns={"index": "Finger-pressure bin (mmHg)", 0: "Trough count"}
    )
    st.table(valid_cnt_df)

    # 2 – Interactive bar chart
    fig_bin = px.bar(
        x=valid_cnt.index.astype(str),
        y=valid_cnt.values,
        labels={"x": "Finger-pressure bin (mmHg)", "y": "N° diastolic troughs"},
        title="Diastolic troughs per 3 mmHg bin"
    )
    st.plotly_chart(fig_bin, use_container_width=True)

    # 3 – Finger-pressure with systolic peaks
    st.subheader("Finger-pressure signal with systolic peaks")
    fig_sys = go.Figure()
    fig_sys.add_trace(go.Scatter(x=time_arr, y=fp_signal,
                                 name="Finger Pressure", line=dict(color="blue")))
    fig_sys.add_trace(go.Scatter(x=time_arr[syst_idx], y=syst_vals,
                                 mode="markers", name="Systolic peaks",
                                 marker=dict(color="red", size=5)))
    fig_sys.update_layout(xaxis_title="Time (s)", yaxis_title="Pressure (mmHg)",
                          template="plotly_white")
    st.plotly_chart(fig_sys, use_container_width=True)

    # 4 – Finger-pressure with detected troughs
    st.subheader("Finger-pressure signal with detected troughs")
    fig_tr = go.Figure()
    fig_tr.add_trace(go.Scatter(x=time_arr, y=fp_signal,
                                name="Finger Pressure", line=dict(color="darkred")))
    fig_tr.add_trace(go.Scatter(x=time_arr[troughs_idx], y=dia_vals,
                                mode="markers", name="Troughs",
                                marker=dict(color="red", size=5)))
    fig_tr.update_layout(xaxis_title="Time (s)", yaxis_title="Pressure (mmHg)",
                         template="plotly_white")
    st.plotly_chart(fig_tr, use_container_width=True)

    # 5 – Blood-pressure variability metrics (organised table)
    st.subheader("Blood-pressure variability metrics")

    def _stats(arr):
        return [arr.mean(), arr.std(),
                arr.std()/arr.mean()*100 if arr.mean() else 0,
                np.abs(np.diff(arr)).mean(),
                arr.min(), arr.max()]

    bp_metrics = pd.DataFrame({
        "Metric": ["Mean (mmHg)", "SD (mmHg)", "CV (%)", "ARV (mmHg)",
                   "Min (mmHg)", "Max (mmHg)"],
        "Finger Pressure":  _stats(fp_signal),
        "Systolic Pressure": _stats(syst_vals),
        "Diastolic Pressure": _stats(dia_vals)
    }).set_index("Metric")

    # nicely formatted & centred
    st.table(bp_metrics.style
             .format("{:.2f}")
             .set_table_styles([
                 {"selector": "th", "props": [("text-align", "center"),
                                              ("font-weight", "bold")]},
                 {"selector": "td", "props": [("text-align", "center")]}
             ]))

    # 6 – All-bin trough counts
    st.subheader("Trough-bin counts (all bins)")
    all_cnt_df = bin_cnts.reset_index().rename(
        columns={"index": "Finger-pressure bin (mmHg)", 0: "Trough count"}
    )
    st.table(all_cnt_df)

    # --- save data-frames for Excel export used in the MSNA tab ------------
    st.session_state["bp_metrics"] = bp_metrics.reset_index()
    st.session_state["bin_counts"] = all_cnt_df

# ───────────────────────────  BAROREFLEX tab ───────────────────────────
with tab_baro:
    st.markdown("## Sympathetic Baroreflex – MSNA incidence vs. diastolic pressure")

    # --- 1. Re-detect cardiac R-waves (same method used in the MSNA tab) ----
    samp_int = float(df["time"].iloc[1] - df["time"].iloc[0])

    def bp_filter(sig, fs=1/samp_int, lo=0.5, hi=45, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
        return filtfilt(b, a, sig)

    ecg_filt = bp_filter(df["ECG"].values)
    peaks_ecg, _ = find_peaks(ecg_filt, distance=int(0.3/samp_int))

    # --- 2. Find diastolic troughs of the finger-pressure signal ----------
    fp_sig  = df["Finger Pressure"].values
    dt      = samp_int
    prom_tr = 1.5
    thr_tr  = fp_sig.mean() - fp_sig.std()
    troughs_idx, _ = find_peaks(-fp_sig, prominence=prom_tr,
                                height=-thr_tr, distance=int(0.7/dt))

    # helper → nearest preceding trough for any index
    def nearest_preceding(idx, trough_idx):
        trough_idx = np.asarray(trough_idx)
        return trough_idx[np.searchsorted(trough_idx, idx, side="right")-1]

    # MSNA bursts already available as filt_idx (from tab_filt)
    msna_bursts = filt_idx

    # --- 3. Build diastolic bins (same 3 mmHg edges as in BP tab) ----------
    bin_edges = np.arange(fp_sig.min(), fp_sig.max() + 3, 3)
    bin_lbls  = [f"{round(b,1)}–{round(b+3,1)}" for b in bin_edges[:-1]]

    # total heart-beats per bin
    hb_trough = fp_sig[nearest_preceding(peaks_ecg, troughs_idx)]
    hb_bins   = pd.cut(hb_trough, bins=bin_edges, labels=bin_lbls,
                       include_lowest=True)
    tot_hb    = hb_bins.value_counts().reindex(bin_lbls, fill_value=0)

    # total sympathetic bursts per bin
    msna_trough = fp_sig[nearest_preceding(msna_bursts, troughs_idx)]
    msna_bins   = pd.cut(msna_trough, bins=bin_edges, labels=bin_lbls,
                         include_lowest=True)
    tot_bursts  = msna_bins.value_counts().reindex(bin_lbls, fill_value=0)

    # incidence (bursts / 100 heart-beats)
    msna_incidence = (tot_bursts / tot_hb).replace([np.inf, np.nan], 0)*100
    valid_mask     = tot_hb > 1                       # ≥ 2 beats in bin
    msna_incidence = msna_incidence[valid_mask]
    tot_hb         = tot_hb[valid_mask]

    # numeric midpoint of each bin (x-axis)
    bin_mid = np.array([(float(l.split("–")[0]) +
                         float(l.split("–")[1]))/2 for l in msna_incidence.index])

    # --- 4. Weighted linear regression ------------------------------------
    wts   = tot_hb.values

    # ── bin-quality check & adaptive fallback ──────────────────────────
    valid_mask = tot_hb > 1          # ≥ 2 heart-beats – preferred rule
    if valid_mask.sum() < 2:         # not enough bins for a fit
        st.warning(
            f"Only {valid_mask.sum()} diastolic-pressure bin(s) contained ≥2 "
            "heart-beats. Falling back to **all bins with at least 1 beat** so "
            "that the baroreflex regression can still be computed."
        )
        valid_mask = tot_hb > 0      # relax to ≥ 1 beat

    # rebuild arrays with (possibly) relaxed mask
    bin_mid_select = bin_mid[valid_mask]
    inc_select     = msna_incidence[valid_mask].values
    wts_select     = tot_hb[valid_mask].values

    # final safeguard – skip if we still have <2 points
    if len(bin_mid_select) < 2:
        st.warning(
            "After all checks there is still fewer than two valid bins – "
            "baroreflex slope cannot be estimated for this recording."
        )
        baro_df = pd.DataFrame({
            "Pressure bin": msna_incidence.index,
            "MSNA incidence (bursts/100 HB)": msna_incidence.values,
            "Heart-beats in bin": tot_hb.values
        })
        st.table(baro_df.style.format({"MSNA incidence (bursts/100 HB)":"{:.2f}"}))
        st.session_state["baro_df"] = None
    else:
        # ── weighted linear regression (unchanged logic) ───────────────
        model = LinearRegression().fit(
            bin_mid_select.reshape(-1, 1),
            inc_select,
            sample_weight=wts_select
        )
        slope = float(model.coef_[0])
        pred  = model.predict(bin_mid_select.reshape(-1, 1))

        # weighted R & R²
        # mx = np.average(bin_mid_select, weights=wts_select)
        # my = np.average(inc_select,   weights=wts_select)
        # cov  = np.sum(wts_select * (bin_mid_select - mx) * (inc_select - my)) / wts_select.sum()
        # stdx = np.sqrt(np.sum(wts_select * (bin_mid_select - mx) ** 2) / wts_select.sum())
        # stdy = np.sqrt(np.sum(wts_select * (inc_select   - my) ** 2) / wts_select.sum())
        r_val = safe_weighted_corr(bin_mid_select, inc_select, wts_select)
        r_sq  = r_val ** 2 if not np.isnan(r_val) else np.nan

        # r_val = cov / (stdx * stdy)
        # r_sq  = r_val ** 2

        df_neural_sel = pd.DataFrame({
                "Diastolic Carotid Diameter Bins": x_sel,
                "MSNA Incidence (Bursts/100 Heartbeats)": y_sel,
                "Total Heartbeats in Bin": wts,
                "Slope":   [slope]   + [""] * (len(x_sel) - 1),
                "Weighted Correlation Coefficient (R)": [r] + [""] * (len(x_sel) - 1),
                "Coefficient of Determination (R²)":    [r2] + [""] * (len(x_sel) - 1),
            })
        st.session_state["neural_arm_sel"] = df_neural_sel  
        
        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Slope (bursts/100 HB per mmHg)", f"{slope:.4f}")
        k2.metric("Weighted R",  f"{r_val:.4f}")
        k3.metric("R²",          f"{r_sq:.4f}")

        # Scatter + fit
        fig_baro = go.Figure()
        fig_baro.add_trace(go.Scatter(x=bin_mid_select, y=inc_select,
                                    mode="markers",
                                    marker=dict(color="darkred", size=8),
                                    name="Incidence"))
        fig_baro.add_trace(go.Scatter(x=bin_mid_select, y=pred,
                                    mode="lines", line=dict(color="red"),
                                    name="Weighted fit"))
        fig_baro.update_layout(title="Baroreflex slope",
                            xaxis_title="Diastolic pressure (bin midpoint, mmHg)",
                            yaxis_title="MSNA incidence (bursts / 100 HB)",
                            template="plotly_white")
        st.plotly_chart(fig_baro, use_container_width=True)

        # Results table & Excel export payload
        baro_df = pd.DataFrame({
            "Pressure bin": msna_incidence.index,
            "MSNA incidence (bursts/100 HB)": msna_incidence.values,
            "Heart-beats in bin": tot_hb.values
        })
        st.table(baro_df.style.format({"MSNA incidence (bursts/100 HB)":"{:.2f}"}))

        st.session_state["baro_df"] = pd.concat(
            [baro_df,
            pd.DataFrame({"Pressure bin":["— OVERALL —"],
                        "MSNA incidence (bursts/100 HB)":[inc_select.mean()],
                        "Heart-beats in bin":[wts_select.sum()],
                        "Slope":[slope],
                        "Weighted R":[r_val],
                        "R²":[r_sq]})],
            ignore_index=True
        )
    
# ───────────────────────  CAROTID DIAMETER TROUGHS tab  ──────────────────────
    with tab_cd:
        st.markdown("## Carotid diameter – trough detection & cleaning")

        # --- detect troughs (as before) ---------------------------------------
        diam_sig = df["Ultrasound - Diameter"].values
        t_arr    = df["time"].values
        dt       = float(df["time"].iloc[1] - df["time"].iloc[0])

        prom_cd, thr_cd = 0.1, diam_sig.mean() - diam_sig.std()
        dist_cd         = int(0.1/dt)
        troughs_cd, _   = find_peaks(-diam_sig, prominence=prom_cd,
                                    height=-thr_cd, distance=dist_cd)

        # ── UI – pick trough numbers to *discard* -----------------------------
        st.metric("Detected troughs", f"{len(troughs_cd)}")
        toss = st.multiselect(
            "❌ Select trough numbers to *remove* (Ctrl/Cmd-click for multi-select):",
            options=list(range(1, len(troughs_cd)+1)),
            help="Numbers correspond to the red-labelled markers in the plot below."
        )
        keep_mask       = ~np.isin(np.arange(1, len(troughs_cd)+1), toss)
        troughs_keep    = troughs_cd[keep_mask]

        # --- master plot with numbered markers --------------------------------
        fig_cd = go.Figure()
        fig_cd.add_scatter(x=t_arr, y=diam_sig, mode="lines",
                        line=dict(color="blue"), name="Carotid diameter")
        fig_cd.add_scatter(x=t_arr[troughs_cd], y=diam_sig[troughs_cd],
                        mode="markers+text", text=[str(i) for i in range(1, len(troughs_cd)+1)],
                        textposition="bottom center", textfont_size=9,
                        marker=dict(color="red", size=6),
                        name="All troughs")
        # highlight *kept* troughs in green
        fig_cd.add_scatter(x=t_arr[troughs_keep], y=diam_sig[troughs_keep],
                        mode="markers", marker=dict(color="green", size=8, symbol="circle"),
                        name="Kept troughs")
        fig_cd.update_layout(template="plotly_white", xaxis_title="Time (s)",
                            yaxis_title="Diameter (a.u.)", xaxis_range=[0, 150])
        st.plotly_chart(fig_cd, use_container_width=True)

        # --- FULL table of detected troughs  ----------------------------------
        st.markdown("### Detected-trough table")
        st.dataframe(                                  # ← new line
            pd.DataFrame({                             #  build the table
                "No.":       np.arange(1, len(troughs_cd)+1),
                "Sample idx":troughs_cd,
                "Time (s)":  t_arr[troughs_cd],
                "Diameter":  diam_sig[troughs_cd]
            }).style.format({"Time (s)":"{:.2f}", "Diameter":"{:.4f}"}),
            height=300, use_container_width=True
        )

        # --- helper: statistics & bin counts ----------------------------------
        def _diam_stats(label, idx):
            vals = diam_sig[idx] if label.startswith("Selected") else diam_sig
            mean, sd  = vals.mean(), vals.std()
            cv        = sd/mean*100 if mean else np.nan
            bins      = np.arange(diam_sig.min(), diam_sig.max()+0.2, 0.2)
            bin_lbls  = [f"{b:.1f}–{b+0.2:.1f}" for b in bins[:-1]]
            bin_counts = pd.cut(vals, bins=bins, labels=bin_lbls, include_lowest=True).value_counts().sort_index()
            return (
                pd.DataFrame({"Metric":["Mean","SD","CV %","Min","Max","N"],
                            label:[mean, sd, cv, vals.min(), vals.max(), len(vals)]}).set_index("Metric"),
                bin_counts[bin_counts>0]     # drop zero bars
            )
        st.session_state["troughs_original"] = troughs_cd      # all detected troughs
        st.session_state["troughs_selected"] = troughs_keep    # user-kept troughs    
        
        stats_orig, bins_orig   = _diam_stats("Original", troughs_cd)
        stats_sel,  bins_sel    = _diam_stats("Selected", troughs_keep)

        # --- side-by-side KPIs -------------------------------------------------
        k1,k2 = st.columns(2)
        k1.subheader("Original");  k1.table(stats_orig.style.format("{:.4f}"))
        k2.subheader("Selected");  k2.table(stats_sel .style.format("{:.4f}"))

        # --- histograms --------------------------------------------------------
        c1,c2 = st.columns(2)
        c1.plotly_chart(px.bar(x=bins_orig.index, y=bins_orig.values,
                            labels={"x":"Diameter bin (mm)","y":"Heart-beats"},
                            title="Original distribution"), use_container_width=True)
        c2.plotly_chart(px.bar(x=bins_sel.index,  y=bins_sel.values,
                            labels={"x":"Diameter bin (mm)","y":"Heart-beats"},
                            title="Selected distribution"), use_container_width=True)

    # --- FULL table of detected troughs  ----------------------------------
        st.markdown("### Detected-trough table")
        st.dataframe(                                  # ← new line
            pd.DataFrame({                             #  build the table
                "No.":       np.arange(1, len(troughs_cd)+1),
                "Sample idx":troughs_cd,
                "Time (s)":  t_arr[troughs_cd],
                "Diameter":  diam_sig[troughs_cd]
            }).style.format({"Time (s)":"{:.2f}", "Diameter":"{:.4f}"}),
            height=300, use_container_width=True
        )
        # --- make data available for the Excel export -------------------------
        st.session_state["diam_stats"] = {
            "orig_stats":  stats_orig.reset_index(),
            "orig_bins":   bins_orig.reset_index().rename(columns={"index":"Diameter bin",0:"Heart-beats"}),
            "sel_stats":   stats_sel.reset_index(),
            "sel_bins":    bins_sel.reset_index().rename(columns={"index":"Diameter bin",0:"Heart-beats"})
        }

# ── 2 ▸ Neural-Arc TAB (cleaned)  ────────────────────────────────────────
with tab_neural:
    st.markdown("## Neural arc – MSNA incidence vs. diastolic **carotid diameter**")

    # ---- prerequisites ---------------------------------------------------
    diam = df["Ultrasound - Diameter"].values

    # data coming from other tabs ↓↓↓
    peaks_ecg_arr = st.session_state.get("peaks_ecg")
    msna_bursts   = st.session_state.get("filt_idx")
    trough_idx    = st.session_state.get("carotid_trough_idx")

    if peaks_ecg_arr is None or msna_bursts is None:
        st.warning("Visit the **Filtered Peaks** and **MSNA** tabs once, "
                   "so the app can detect bursts and R-waves.")
        st.stop()

    if trough_idx is None or len(trough_idx) < 3:
        st.warning("No carotid-diameter troughs available – run the "
                   "**Carotid Diameter Troughs** tab first.")
        st.stop()

    # ---- helper: nearest preceding trough --------------------------------
    def _prev(idx, troughs):
        troughs = np.asarray(troughs)
        pos = np.searchsorted(troughs, idx, side="right") - 1
        return troughs[max(pos, 0)]                 # ← protects against −1

    # ---- diameter bins ---------------------------------------------------
    step   = 0.2
    edges  = np.arange(diam.min(), diam.max() + step, step)
    labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges)-1)]

    hb_trough   = diam[_prev(peaks_ecg_arr, trough_idx)]
    msna_trough = diam[_prev(msna_bursts,   trough_idx)]

    hb_bins   = pd.cut(hb_trough,   bins=edges, labels=labels, include_lowest=True)
    msna_bins = pd.cut(msna_trough, bins=edges, labels=labels, include_lowest=True)

    hb_counts   = hb_bins.value_counts().reindex(labels, fill_value=0)
    msna_counts = msna_bins.value_counts().reindex(labels, fill_value=0)

    # remove sparse bins (≤2 HB)
    valid_mask = hb_counts > 2
    inc        = (msna_counts / hb_counts.replace(0, np.nan) * 100).replace([np.inf, np.nan], 0)

    x_mid = np.array([(float(l.split('-')[0]) + float(l.split('-')[1])) / 2 for l in labels])
    x_sel, y_sel, wts = x_mid[valid_mask], inc[valid_mask].values, hb_counts[valid_mask].values

    if len(x_sel) < 2:
        st.warning("Fewer than two diameter bins with >2 heart-beats – regression skipped.")
        st.stop()

    # ---- weighted regression --------------------------------------------
    mdl   = LinearRegression().fit(x_sel.reshape(-1, 1), y_sel, sample_weight=wts)
    slope = float(mdl.coef_[0])
    pred  = mdl.predict(x_sel.reshape(-1, 1))

    # weighted correlation
    mx, my = np.average(x_sel, weights=wts), np.average(y_sel, weights=wts)
    cov    = np.sum(wts * (x_sel - mx) * (y_sel - my)) / wts.sum()
    # weighted correlation
    r  = safe_weighted_corr(x_sel, y_sel, wts)
    r2 = r ** 2 if not np.isnan(r) else np.nan
    # r      = cov / (np.sqrt(np.sum(wts * (x_sel - mx) ** 2) / wts.sum())
    #                * np.sqrt(np.sum(wts * (y_sel - my) ** 2) / wts.sum()))
    # r2     = r ** 2

    # ---- KPIs ------------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Slope (bursts/100 HB · mm⁻¹)", f"{slope:.4f}")
    c2.metric("Weighted R", f"{r:.4f}")
    c3.metric("R²",         f"{r2:.4f}")

    # ---- plot ------------------------------------------------------------
    fig = go.Figure()
    fig.add_scatter(x=x_sel, y=y_sel, mode="markers",
                    marker=dict(color="navy", size=7), name="Incidence")
    fig.add_scatter(x=x_sel, y=pred, mode="lines",
                    line=dict(color="navy", dash="dash"), name="Weighted fit")
    fig.update_layout(template="plotly_white",
                      title="Neural-arc slope (diameter bins)",
                      xaxis_title="Diastolic carotid diameter (mm, bin midpoint)",
                      yaxis_title="MSNA incidence (bursts / 100 HB)")
    st.plotly_chart(fig, use_container_width=True)

    # ---- incidence table & export payload --------------------------------
    res_df = pd.DataFrame({
        "Diameter bin": labels,
        "Heart-beats": hb_counts.values,
        "MSNA bursts": msna_counts.values,
        "Incidence":   inc.values
    })
    st.subheader("Incidence table (all bins)")
    st.dataframe(res_df.style.format({"Incidence": "{:.2f}"}), height=320)

    st.session_state["neural_arc_df"] = pd.concat(
        [res_df,
         pd.DataFrame({"Diameter bin": ["— OVERALL —"],
                       "Heart-beats":  [hb_counts.sum()],
                       "MSNA bursts":  [msna_counts.sum()],
                       "Incidence":    [inc[valid_mask].mean()],
                       "Slope":        [slope],
                       "Weighted R":   [r],
                       "R²":           [r2]})],
        ignore_index=True
    )

# ─────────────────────  Neural-Arc  ➜  Original VS Selected  ───────────────
with tab_neural_cmp:
    st.markdown("## Neural arc – *original* vs *selected* carotid troughs")

    # ---- pull data produced by other tabs --------------------------------
    peaks_ecg_arr   = st.session_state.get("peaks_ecg")
    msna_bursts     = st.session_state.get("filt_idx")
    troughs_orig    = st.session_state.get("troughs_original")
    troughs_sel     = st.session_state.get("troughs_selected")

    # guard-clauses (same style you use elsewhere)
    if None in (peaks_ecg_arr, msna_bursts, troughs_orig, troughs_sel):
        st.info("Make sure **Filtered Peaks**, **MSNA** and **Carotid Diameter "
                "Troughs** tabs have been visited once.")
        st.stop()

    if len(troughs_sel) < 3:
        st.warning("The *selected* trough set contains fewer than three points – "
                   "analysis skipped.")
        st.stop()

    # ---- helper ----------------------------------------------------------
    def _prev(event_idx, trough_idx):
        trough_idx = np.asarray(trough_idx)
        pos = np.searchsorted(trough_idx, event_idx, side="right") - 1
        return trough_idx[max(pos, 0)]

    diam = df["Ultrasound - Diameter"].values
    step = 0.2
    edges  = np.arange(diam.min(), diam.max() + step, step)
    labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges)-1)]
    midpts = np.array([(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)])

    def _incidence(troughs):
        hb = diam[_prev(peaks_ecg_arr, troughs)]
        sb = diam[_prev(msna_bursts,     troughs)]
        hb_bins = pd.cut(hb, bins=edges, labels=labels, include_lowest=True)
        sb_bins = pd.cut(sb, bins=edges, labels=labels, include_lowest=True)
        hb_cnt  = hb_bins.value_counts().reindex(labels, fill_value=0)
        sb_cnt  = sb_bins.value_counts().reindex(labels, fill_value=0)
        inc     = (sb_cnt / hb_cnt.replace(0, np.nan) * 100).replace([np.inf, np.nan], 0)
        return hb_cnt, inc

    hb_o, inc_o = _incidence(troughs_orig)
    hb_s, inc_s = _incidence(troughs_sel)

    # keep bins with >2 HB (orig) and >3 HB (sel) to match the standalone script
    msk_o = hb_o > 2
    msk_s = hb_s > 3

    x_o, y_o, w_o = midpts[msk_o], inc_o[msk_o].values, hb_o[msk_o].values
    x_s, y_s, w_s = midpts[msk_s], inc_s[msk_s].values, hb_s[msk_s].values

    # ---- regressions -----------------------------------------------------
    mdl_o = LinearRegression().fit(x_o.reshape(-1,1), y_o, sample_weight=w_o)
    mdl_s = LinearRegression().fit(x_s.reshape(-1,1), y_s, sample_weight=w_s)

    slope_o, slope_s = float(mdl_o.coef_[0]), float(mdl_s.coef_[0])
    r_o = safe_weighted_corr(x_o, y_o)  
    r2_o = r_o**2
    r_s = safe_weighted_corr(x_s, y_s)
    r2_s = r_s**2

    # ---- Excel-ready dataframe for ORIGINAL set -------------------
    df_neural_orig = pd.DataFrame({
        "Diastolic Carotid Diameter Bins": x_o,
        "MSNA Incidence (Bursts/100 Heartbeats)": y_o,
        "Total Heartbeats in Bin": w_o,
        "Slope":   [slope_o] + [""] * (len(x_o) - 1),
        "Weighted Correlation Coefficient (R)": [r_o] + [""] * (len(x_o) - 1),
        "Coefficient of Determination (R²)":    [r2_o] + [""] * (len(x_o) - 1),
    })
    st.session_state["neural_arm_orig"] = df_neural_orig

    # ---- KPI row ---------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Slope orig", f"{slope_o:.4f}")
    k2.metric("R orig",     f"{r_o:.4f}")
    k3.metric("Slope sel",  f"{slope_s:.4f}")
    k4.metric("R sel",      f"{r_s:.4f}")

    # ---- plot ------------------------------------------------------------
    fig = go.Figure()
    fig.add_scatter(x=x_o, y=y_o, mode="markers",
                    marker=dict(color="steelblue", size=7), name="Original")
    fig.add_scatter(x=x_o, y=mdl_o.predict(x_o.reshape(-1,1)),
                    mode="lines", line=dict(color="steelblue", dash="dash"),
                    name="Orig fit")
    fig.add_scatter(x=x_s, y=y_s, mode="markers",
                    marker=dict(color="firebrick", size=7), name="Selected")
    fig.add_scatter(x=x_s, y=mdl_s.predict(x_s.reshape(-1,1)),
                    mode="lines", line=dict(color="firebrick", dash="dash"),
                    name="Sel fit")
    fig.update_layout(template="plotly_white",
                      xaxis_title="Diastolic carotid diameter (mm, bin midpoint)",
                      yaxis_title="MSNA incidence (bursts / 100 HB)",
                      title="Neural-Arc: Original vs Selected troughs")
    st.plotly_chart(fig, use_container_width=True)

    # ---- excluded diameters list ----------------------------------------
    excl_idx = set(troughs_orig) - set(troughs_sel)
    excl_diam = diam[list(excl_idx)]
    if len(excl_diam):
        st.markdown("### Excluded diameters")
        st.write(pd.Series(excl_diam).to_frame("Diameter").style.format("{:.4f}"))

    # ---- results table & export payload ----------------------------------
    df_cmp = pd.DataFrame({
        "Bin": labels,
        "HB orig": hb_o.values,
        "Inc orig": inc_o.values,
        "HB sel":  hb_s.values,
        "Inc sel": inc_s.values
    })
    st.subheader("Incidence table (all bins)")
    st.dataframe(df_cmp.style.format({"Inc orig":"{:.2f}", "Inc sel":"{:.2f}"}),
                 height=340)

    # store for Excel export
    st.session_state["neural_cmp_df"] = pd.concat(
        [df_cmp,
         pd.DataFrame({"Bin":["— OVERALL —"],
                       "HB orig":[hb_o.sum()],
                       "Inc orig":[inc_o[msk_o].mean()],
                       "HB sel":[hb_s.sum()],
                       "Inc sel":[inc_s[msk_s].mean()],
                       "Slope orig":[slope_o], "R orig":[r_o], "R² orig":[r2_o],
                       "Slope sel":[slope_s], "R sel":[r_s], "R² sel":[r2_s]})],
        ignore_index=True
    )
# ────────────────────  MECHANICAL ARC tab  ────────────────────────────────
with tab_mecharc:
    st.markdown("## Mechanical arc – mean carotid diameter vs. diastolic pressure")

    # prerequisites --------------------------------------------------------
    peaks_ecg_arr = st.session_state.get("peaks_ecg")
    if peaks_ecg_arr is None or len(peaks_ecg_arr) < 3:
        st.info("Visit the **MSNA** tab once so R-waves are detected first.")
        st.stop()

    fp  = df["Finger Pressure"].values
    cd  = df["Ultrasound - Diameter"].values

    # 1. diastolic pressure per cardiac cycle -----------------------------
    dia_p = np.array(
        [fp[peaks_ecg_arr[i]:peaks_ecg_arr[i + 1]].min()
         for i in range(len(peaks_ecg_arr) - 1)],
        dtype=float
    )
    diam_at_r = cd[peaks_ecg_arr[:-1]]          # carotid diameter at R-wave
    total_hb  = len(peaks_ecg_arr)

    # 2. 3-mmHg bins -------------------------------------------------------
    bin_edges  = np.arange(dia_p.min(), dia_p.max() + 3, 3)
    bin_labels = [f"{round(b,1)}–{round(b+3,1)}" for b in bin_edges[:-1]]

    df_mech = pd.DataFrame({
        "Diastolic_Pressure":  dia_p,
        "Carotid_Diameter":    diam_at_r
    })
    df_mech["Pressure_Bin"] = pd.cut(df_mech["Diastolic_Pressure"],
                                     bins=bin_edges, labels=bin_labels,
                                     include_lowest=True)

    mean_diam = df_mech.groupby("Pressure_Bin")["Carotid_Diameter"]\
                       .mean().reindex(bin_labels)
    hb_counts = df_mech["Pressure_Bin"].value_counts()\
                                      .reindex(bin_labels, fill_value=0)

    valid = hb_counts > 3              # > 3 heart-beats, as requested
    if valid.sum() < 2:
        st.warning("Fewer than two pressure bins contain > 3 beats – regression skipped.")
        st.stop()

    x_mid   = np.array([(float(l.split('-')[0]) + float(l.split('-')[1])) / 2
                        for l in mean_diam.index])[valid]
    y_mean  = mean_diam[valid].values
    wts     = hb_counts[valid].values

    # 3. weighted regression ----------------------------------------------
    mdl = LinearRegression().fit(x_mid.reshape(-1, 1), y_mean, sample_weight=wts)
    slope, intercept = float(mdl.coef_[0]), float(mdl.intercept_)
    r     = safe_weighted_corr(x_mid, y_mean, wts)
    r2    = r**2 if not np.isnan(r) else np.nan
    y_hat = mdl.predict(x_mid.reshape(-1, 1))

    # 4. KPIs --------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total HB",              f"{total_hb}")
    k2.metric("Slope (mm ⁻¹)",         f"{slope:.4f}")
    k3.metric("Weighted R",            f"{r:.4f}")
    k4.metric("R²",                    f"{r2:.4f}")

    # 5. scatter + fit -----------------------------------------------------
    fig_mech = go.Figure()
    fig_mech.add_scatter(x=x_mid, y=y_mean, mode="markers",
                         marker=dict(color="darkgreen", size=wts),
                         name="Mean diameter / bin")
    fig_mech.add_scatter(x=x_mid, y=y_hat, mode="lines",
                         line=dict(color="darkgreen", dash="dash"),
                         name="Weighted fit")
    fig_mech.update_layout(template="plotly_white",
                           title="Mechanical component of the carotid baroreflex",
                           xaxis_title="Diastolic pressure (bin midpoint, mmHg)",
                           yaxis_title="Mean carotid diameter (mm)")
    st.plotly_chart(fig_mech, use_container_width=True)

    # 6. results table -----------------------------------------------------
    mech_df = pd.DataFrame({
        "Pressure Bin":       mean_diam.index,
        "Mean Diameter (mm)": mean_diam.values,
        "Heart-beats":        hb_counts.values
    })
    st.subheader("Per-bin summary")
    st.dataframe(mech_df.style.format({"Mean Diameter (mm)": "{:.3f}"}), height=280)

    # 7. store Excel-ready dataframe --------------------------------------
    mech_export = pd.DataFrame({
        "Diastolic Pressure Bin": x_mid,
        "Mean Carotid Diameter":  y_mean,
        "Total Heartbeats in Bin": wts,
        "Slope":        [slope] + [""] * (len(x_mid) - 1),
        "Intercept":    [intercept] + [""] * (len(x_mid) - 1),
        "Weighted Correlation Coefficient (R)": [r] + [""] * (len(x_mid) - 1),
        "Coefficient of Determination (R²)":    [r2] + [""] * (len(x_mid) - 1),
    })
    st.session_state["mech_arc_df"] = mech_export

# ───────────────────  MECHANICAL ARC  ➜  *SELECTED* troughs  ──────────────────
with tab_mecharc_sel:
    st.markdown("## Mechanical arc – **selected** carotid troughs")

    # prerequisites --------------------------------------------------------
    peaks_ecg_arr   = st.session_state.get("peaks_ecg")
    troughs_sel     = st.session_state.get("troughs_selected")   # from Carotid tab

    if peaks_ecg_arr is None or troughs_sel is None or len(troughs_sel) < 4:
        st.info("Make sure the **MSNA** and **Carotid Diameter Troughs** tabs "
                "have been visited and that ≥ 4 troughs are selected.")
        st.stop()

    fp, cd = df["Finger Pressure"].values, df["Ultrasound - Diameter"].values

    # helper – nearest preceding R-wave for each trough --------------------
    troughs_sel = np.asarray(troughs_sel, dtype=int)
    precedente  = np.searchsorted(peaks_ecg_arr, troughs_sel, side="right") - 1
    r_idx_for_trough = peaks_ecg_arr[precedente]

    # diastolic pressure between consecutive R-waves -----------------------
    dia_p_sel = np.array(
        [fp[r_idx_for_trough[i]: r_idx_for_trough[i+1]].min()
         if i+1 < len(r_idx_for_trough) else fp[r_idx_for_trough[i]:].min()
         for i in range(len(r_idx_for_trough))],
        dtype=float
    )
    dia_diam  = cd[troughs_sel]
    total_sel = len(troughs_sel)

    # 3-mmHg bins ----------------------------------------------------------
    bin_edges  = np.arange(dia_p_sel.min(), dia_p_sel.max() + 3, 3)
    bin_labels = [f"{round(b,1)}–{round(b+3,1)}" for b in bin_edges[:-1]]

    df_sel = pd.DataFrame({
        "DiaP": dia_p_sel,
        "DiaDiam": dia_diam
    })
    df_sel["Bin"] = pd.cut(df_sel["DiaP"], bins=bin_edges,
                           labels=bin_labels, include_lowest=True)

    mean_diam = df_sel.groupby("Bin")["DiaDiam"].mean().reindex(bin_labels)
    n_trough  = df_sel["Bin"].value_counts().reindex(bin_labels, fill_value=0)

    valid = n_trough > 3              # > 3 troughs rule
    if valid.sum() < 2:
        st.warning("Fewer than two pressure bins contain > 3 selected troughs – "
                   "regression skipped.")
        st.stop()

    x_mid = np.array([(float(l.split('-')[0]) + float(l.split('-')[1]))/2
                      for l in mean_diam.index])[valid]
    y_mid = mean_diam[valid].values
    wts   = n_trough[valid].values

    # weighted regression --------------------------------------------------
    mdl = LinearRegression().fit(x_mid.reshape(-1,1), y_mid, sample_weight=wts)
    slope_s, intercept_s = float(mdl.coef_[0]), float(mdl.intercept_)
    r_s  = safe_weighted_corr(x_mid, y_mid, wts)
    r2_s = r_s**2 if not np.isnan(r_s) else np.nan
    y_hat = mdl.predict(x_mid.reshape(-1,1))

    # KPIs -----------------------------------------------------------------
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Selected troughs",     f"{total_sel}")
    c2.metric("Slope (mm ⁻¹)",        f"{slope_s:.4f}")
    c3.metric("Weighted R",           f"{r_s:.4f}")
    c4.metric("R²",                   f"{r2_s:.4f}")

    # plot -----------------------------------------------------------------
    fig_ms = go.Figure()
    fig_ms.add_scatter(x=x_mid, y=y_mid, mode="markers",
                       marker=dict(color="firebrick", size=wts),
                       name="Mean diameter / bin")
    fig_ms.add_scatter(x=x_mid, y=y_hat, mode="lines",
                       line=dict(color="firebrick", dash="dash"),
                       name="Weighted fit")
    fig_ms.update_layout(template="plotly_white",
                         title="Mechanical baroreflex – selected troughs",
                         xaxis_title="Diastolic pressure (bin midpoint, mmHg)",
                         yaxis_title="Mean carotid diameter (mm)")
    st.plotly_chart(fig_ms, use_container_width=True)

    # table ---------------------------------------------------------------
    mecs_df = pd.DataFrame({
        "Pressure Bin": bin_labels,
        "Mean Diameter (mm)": mean_diam.values,
        "Selected Troughs": n_trough.values
    })
    st.subheader("Per-bin summary")
    st.dataframe(mecs_df.style.format({"Mean Diameter (mm)": "{:.3f}"}), height=280)

    # Excel payload -------------------------------------------------------
    mech_sel_export = pd.DataFrame({
        "Diastolic Pressure Bin": x_mid,
        "Mean Carotid Diameter":  y_mid,
        "Selected Troughs in Bin": wts,
        "Slope":        [slope_s] + [""] * (len(x_mid) - 1),
        "Intercept":    [intercept_s] + [""] * (len(x_mid) - 1),
        "Weighted Correlation Coefficient (R)": [r_s] + [""] * (len(x_mid) - 1),
        "Coefficient of Determination (R²)":    [r2_s] + [""] * (len(x_mid) - 1),
    })
    st.session_state["mech_arc_sel_df"] = mech_sel_export
