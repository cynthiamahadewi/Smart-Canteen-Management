"""
Smart Canteen Management — Streamlit version
Clean layout using st.sidebar for navigation, custom CSS for Flask-matching colors.

Run with:
    source .venv/bin/activate
    streamlit run app_streamlit.py
"""

import os
import sys
import base64
import tempfile
import copy
import json

import streamlit as st
import streamlit.components.v1 as components

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Canteen Management System",
    page_icon="🍽️",
    layout="wide",
)

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_HANDOFF = os.path.join(BASE_DIR, "Model_Handoff")
sys.path.insert(0, MODEL_HANDOFF)

import dataset
# dataset.EXCEL_PATH = r"/Users/cynthiaathena/Documents/S2/School/Spring/242B/Project/food detection/Model_Handoff/data_original.xlsx"
dataset.EXCEL_PATH = r"Model_Handoff/data_original.xlsx"

from predict import load_model, predict as m5_predict
from Serving_Size_Optimization.optimization_engine import run_serving_optimization

# ── Constants ─────────────────────────────────────────────────────────────────
DISHES_DATA = {
    'Orek Tempe':        {'P': 4000,  'C': 15, 'S_base': 29.59,  'D_base': 22, 'R_i': 0.61, 'alpha': 0.10},
    'Bali Tahu':         {'P': 4000,  'C': 12, 'S_base': 34.64,  'D_base': 22, 'R_i': 0.47, 'alpha': 0.8},
    'Telur Mata Sapi':   {'P': 6000,  'C': 45, 'S_base': 47.75,  'D_base': 20, 'R_i': 0.75, 'alpha': 0.8},
    'Ayam bumbu bistik': {'P': 15000, 'C': 75, 'S_base': 55.32,  'D_base': 22, 'R_i': 0.67, 'alpha': 1.6},
    'Ayam laos':         {'P': 15000, 'C': 85, 'S_base': 52.25,  'D_base': 16, 'R_i': 0.30, 'alpha': 1.6},
    'Rendang':           {'P': 18000, 'C': 95, 'S_base': 54.39,  'D_base': 18, 'R_i': 0.54, 'alpha': 1.6},
    'Bali Telur':        {'P': 7000,  'C': 48, 'S_base': 56.10,  'D_base': 21, 'R_i': 0.73, 'alpha': 0.8},
    'Ikan Acar Kuning':  {'P': 16000, 'C': 80, 'S_base': 39.00,  'D_base': 20, 'R_i': 0.53, 'alpha': 1.6},
    'Nasi':              {'P': 5000,  'C': 6,  'S_base': 146.32, 'D_base': 78, 'R_i': 0.62, 'alpha': 0.10},
}
PRESET_MENUS = [1, 7, 9, 11, 12, 13, 15, 23, 31]
CKPT = os.path.join(BASE_DIR, "Model_Handoff", "ckpt_M5.pt")

# ── Session state ─────────────────────────────────────────────────────────────
if "dishes" not in st.session_state:
    st.session_state.dishes = copy.deepcopy(DISHES_DATA)
if "page" not in st.session_state:
    st.session_state.page = "plate"

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model("M5", CKPT)

def find_dish_key(raw_name, dishes):
    if raw_name in dishes: return raw_name
    raw_lower = raw_name.lower()
    for key in dishes:
        if key.lower() == raw_lower: return key
    raw_prefix = raw_name.split("_")[0]
    for key in dishes:
        if key.split("_")[0] == raw_prefix: return key
    for key in dishes:
        if raw_lower in key.lower() or key.lower() in raw_lower: return key
    return None

def img_to_b64(path):
    try:
        with open(path, "rb") as f:
            ext = os.path.splitext(path)[-1].lstrip(".") or "jpeg"
            return f"data:image/{ext};base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""

def uploaded_to_b64(uploaded_file):
    data = uploaded_file.read()
    uploaded_file.seek(0)
    ext = os.path.splitext(uploaded_file.name)[-1].lstrip(".") or "jpeg"
    return f"data:image/{ext};base64," + base64.b64encode(data).decode()

# ── Banner image ──────────────────────────────────────────────────────────────
banner_path = os.path.join(BASE_DIR, "static", "images", "food_bg.jpeg")
banner_b64  = img_to_b64(banner_path)
banner_css  = f'url("{banner_b64}")' if banner_b64 else "linear-gradient(135deg,#0d4f47,#1a7a6e)"

# ── Global CSS — clean, no fighting with Streamlit layout ────────────────────
st.markdown(f"""
<style>
/* ── Hide default Streamlit chrome ── */
#MainMenu, footer {{ display: none !important; }}
header[data-testid="stHeader"] {{ display: none !important; }}

/* ── Page background ── */
.stApp {{ background: #f0f2f5 !important; }}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {{
    background: #0d4f47 !important;
    min-width: 220px !important;
    max-width: 220px !important;
}}
[data-testid="stSidebar"] * {{ color: #fff !important; }}
[data-testid="stSidebarContent"] {{ padding: 0 !important; }}

/* ── Sidebar brand block ── */
.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 22px 18px;
    border-bottom: 0.5px solid rgba(255,255,255,.15);
    margin-bottom: 8px;
}}
.sidebar-logo {{
    width: 36px; height: 36px;
    background: #f5a623;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 18px; color: #fff !important;
    flex-shrink: 0;
}}
.sidebar-brand-name {{ font-size: 15px; font-weight: 700; line-height: 1.2; }}
.sidebar-brand-sub  {{ font-size: 10px; opacity: 0.55; }}

/* ── Sidebar nav buttons ── */
[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    text-align: left;
    background: transparent !important;
    border: none !important;
    color: rgba(255,255,255,0.75) !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    margin-bottom: 2px;
    transition: background 0.15s;
    box-shadow: none !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,255,255,0.1) !important;
    color: #fff !important;
}}
.nav-active > div > button {{
    background: rgba(255,255,255,0.18) !important;
    color: #fff !important;
    font-weight: 600 !important;
}}

/* ── Remove extra sidebar padding ── */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0 !important;
    padding: 0 10px;
}}

/* ── Main content padding ── */
[data-testid="block-container"] {{
    padding: 0 !important;
    max-width: 100% !important;
}}
.main .block-container {{
    padding: 0 !important;
}}

/* ── Banner ── */
.cs-banner {{
    width: 100%;
    height: 160px;
    background-image: {banner_css};
    background-size: cover;
    background-position: center;
    position: relative;
    margin-bottom: 20px;
}}
.cs-banner-overlay {{
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, rgba(13,79,71,0.80) 0%, rgba(13,79,71,0.15) 100%);
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 0 32px;
}}
.cs-banner-title {{
    font-size: 24px; font-weight: 700;
    color: #fff; margin-bottom: 4px;
    font-family: system-ui, -apple-system, sans-serif;
}}
.cs-banner-sub {{
    font-size: 13px; color: rgba(255,255,255,0.85);
    font-family: system-ui, -apple-system, sans-serif;
}}

/* ── Cards ── */
.cs-card {{
    background: #fff;
    border-radius: 12px;
    padding: 20px;
    border: 0.5px solid #e0ddd6;
    height: 100%;
}}
.cs-card-title {{
    font-size: 11px; font-weight: 600;
    letter-spacing: .07em; text-transform: uppercase;
    color: #5f5e5a; margin-bottom: 16px;
    display: flex; align-items: center; gap: 7px;
    font-family: system-ui, -apple-system, sans-serif;
}}
.dot-blue  {{ width:7px;height:7px;border-radius:50%;background:#378ADD;display:inline-block;flex-shrink:0; }}
.dot-green {{ width:7px;height:7px;border-radius:50%;background:#1d9e75;display:inline-block;flex-shrink:0; }}

/* ── Result widgets ── */
.ratio-big {{ font-size: 38px; font-weight: 700; color: #2c2c2a; line-height: 1; }}
.bin-pill {{
    display: inline-block; padding: 4px 12px; border-radius: 10px;
    font-size: 12px; background: #e1f5ee; color: #085041;
    font-weight: 500; vertical-align: middle; margin-left: 10px;
}}
.progress-track {{
    height: 8px; background: #f1efe8; border-radius: 4px;
    overflow: hidden; margin: 12px 0 4px;
}}
.progress-fill {{ height: 100%; background: #0d4f47; border-radius: 4px; }}
.progress-labels {{
    display: flex; justify-content: space-between;
    font-size: 10px; color: #b4b2a9;
}}
.section-label {{
    font-size: 11px; font-weight: 600; color: #888780;
    letter-spacing: .06em; text-transform: uppercase;
    margin: 20px 0 10px;
    font-family: system-ui, -apple-system, sans-serif;
}}
.coral-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; }}
.coral-card {{
    background: #f7f6f2; border-radius: 8px;
    padding: 10px; text-align: center;
}}
.coral-label {{ font-size: 10px; color: #888780; margin-bottom: 4px; }}
.coral-val   {{ font-size: 17px; font-weight: 600; color: #2c2c2a; }}
.coral-bar   {{ height: 3px; background: #e8e6e0; border-radius: 2px; overflow: hidden; margin-top: 6px; }}
.coral-fill  {{ height: 100%; background: #0d4f47; border-radius: 2px; }}
.top3-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 0.5px solid #f1efe8; font-size: 13px;
    font-family: system-ui, -apple-system, sans-serif;
}}
.top3-row:last-child {{ border-bottom: none; }}
.top3-bar  {{ width: 70px; height: 5px; background: #f1efe8; border-radius: 3px; overflow: hidden; display: inline-block; }}
.top3-fill {{ height: 100%; background: #f5a623; border-radius: 3px; }}
.top3-pct  {{ font-size: 11px; color: #888780; min-width: 36px; text-align: right; }}
.idle-msg  {{ text-align: center; color: #aaa; padding: 48px 0; font-size: 13px; }}

/* ── Preview image ── */
.preview-img {{
    width: 100%; height: 190px; object-fit: cover;
    border-radius: 9px; display: block; margin-top: 10px;
}}
.preview-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }}
.preview-item img {{
    width: 100%; height: 120px; object-fit: cover;
    border-radius: 8px; border: 0.5px solid #d3d1c7; display: block;
}}
.preview-label {{ font-size: 10px; color: #888780; margin-bottom: 4px; }}

/* ── Optimization table ── */
.opt-table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; margin-top: 8px; }}
.opt-table thead tr {{ background: #5d4037; color: #fff; }}
.opt-table th, .opt-table td {{
    padding: 10px 14px; text-align: left; border-bottom: 1px solid #e0d7c6;
    font-family: system-ui, -apple-system, sans-serif;
}}
.opt-table tbody tr:hover {{ background: #fdf5e6; }}
.optimized-val {{ color: #2e7d32; font-weight: 700; font-size: 1.05em; }}
.status-ok   {{ color: #1d9e75; font-size: 12px; padding: 8px 0; }}
.status-warn {{ color: #d4891a; font-size: 12px; padding: 8px 0; }}

/* ── File uploader ── */
div[data-testid="stFileUploadDropzone"] {{
    background: #fafaf8 !important;
    border: 1.5px dashed #b4b2a9 !important;
    border-radius: 10px !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: transparent;
    border-bottom: none !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: #f7f6f2 !important;
    border-radius: 8px !important;
    border: 0.5px solid #d3d1c7 !important;
    color: #5f5e5a !important;
    font-size: 12px !important;
    padding: 7px 16px !important;
}}
.stTabs [aria-selected="true"] {{
    background: #0d4f47 !important;
    color: #fff !important;
    border-color: #0d4f47 !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-border"]    {{ display: none !important; }}

/* ── Main action buttons — orange ── */
.main .stButton > button,
[data-testid="block-container"] .stButton > button {{
    background: #f5a623 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 0 !important;
    box-shadow: none !important;
}}
.main .stButton > button:hover,
[data-testid="block-container"] .stButton > button:hover {{
    background: #d4891a !important;
}}

/* Reset button — smaller, subtle */
.reset-btn > div > button {{
    background: #f7f6f2 !important;
    color: #5f5e5a !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    border: 0.5px solid #d3d1c7 !important;
    padding: 6px 0 !important;
}}

/* ── Column gap ── */
[data-testid="stHorizontalBlock"] {{
    gap: 16px;
    padding: 0 20px 20px;
    align-items: start;
}}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-logo">C</div>
      <div>
        <div class="sidebar-brand-name">CANTEEN</div>
        <div class="sidebar-brand-sub">Management System</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    plate_cls = "nav-active" if st.session_state.page == "plate" else ""
    opt_cls   = "nav-active" if st.session_state.page == "opt"   else ""

    with st.container():
        st.markdown(f'<div class="{plate_cls}">', unsafe_allow_html=True)
        if st.button("⊞  Plate Analysis", key="nav_plate"):
            st.session_state.page = "plate"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown(f'<div class="{opt_cls}">', unsafe_allow_html=True)
        if st.button("⊟  Optimization", key="nav_opt"):
            st.session_state.page = "opt"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ── Banner ────────────────────────────────────────────────────────────────────
sub_text = (
    "AI-powered plate consumption analysis"
    if st.session_state.page == "plate"
    else "AI-powered serving size optimization"
)
st.markdown(f"""
<div class="cs-banner">
  <div class="cs-banner-overlay">
    <div class="cs-banner-title">Smart Canteen</div>
    <div class="cs-banner-sub">{sub_text}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PLATE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "plate":

    left, right = st.columns([1.1, 1.9])

    # ── LEFT CARD ────────────────────────────────────────────────────────
    with left:
        st.markdown('<div class="cs-card"><div class="cs-card-title"><span class="dot-blue"></span> Plate Photo Upload</div>', unsafe_allow_html=True)

        img_path = None
        tab_up, tab_pre = st.tabs(["📤 Upload photo", "🖼 Use preset"])

        with tab_up:
            uploaded = st.file_uploader(
                "Drop a plate photo here",
                type=["jpg", "jpeg", "png"],
                key="plate_upload",
                label_visibility="collapsed",
            )
            if uploaded:
                b64 = uploaded_to_b64(uploaded)
                st.markdown(f'<img src="{b64}" class="preview-img">', unsafe_allow_html=True)
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded.name)[-1] or ".jpg"
                ) as tmp:
                    uploaded.seek(0)
                    tmp.write(uploaded.read())
                    img_path = tmp.name

        with tab_pre:
            choice   = st.selectbox("Select a menu", [f"menu{n}" for n in PRESET_MENUS], key="preset_sel", label_visibility="collapsed")
            after_p  = os.path.join(BASE_DIR, "static", "presets", f"{choice}.jpg")
            before_p = os.path.join(BASE_DIR, "static", "presets", f"{choice}_before.jpg")
            bb, ba   = img_to_b64(before_p), img_to_b64(after_p)
            if bb or ba:
                before_html = f"<img src='{bb}'>" if bb else "<div style='height:120px;background:#eee;border-radius:8px'></div>"
                after_html  = f"<img src='{ba}'>" if ba else "<div style='height:120px;background:#eee;border-radius:8px'></div>"
                st.markdown(f"""
                <div class="preview-pair">
                  <div class="preview-item"><div class="preview-label">Before meal</div>{before_html}</div>
                  <div class="preview-item"><div class="preview-label">After meal</div>{after_html}</div>
                </div>
                """, unsafe_allow_html=True)
            if os.path.exists(after_p):
                img_path = after_p

        run_btn = st.button("Analyse this plate", key="run_plate", disabled=(img_path is None), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT CARD ───────────────────────────────────────────────────────
    with right:
        st.markdown('<div class="cs-card"><div class="cs-card-title"><span class="dot-green"></span> M5 Model Output</div>', unsafe_allow_html=True)

        if run_btn and img_path:
            with st.spinner("Running M5 model…"):
                try:
                    model  = get_model()
                    result = m5_predict(model, "M5", img_path)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            pred_r = result["pred_r"]
            pct    = int(pred_r * 100)
            bin_lbl = result["bin"]

            st.markdown(f"""
            <div class="ratio-big">{pct}%<span class="bin-pill">{bin_lbl}</span></div>
            <div class="progress-track"><div class="progress-fill" style="width:{pct}%"></div></div>
            <div class="progress-labels"><span>0%</span><span>100%</span></div>
            """, unsafe_allow_html=True)

            # CORAL thresholds
            if result.get("threshold_str"):
                thresholds = []
                for p in result["threshold_str"].split(" "):
                    if "=" in p:
                        lbl, v = p.split("=", 1)
                        try:
                            thresholds.append((lbl.strip(), float(v.strip())))
                        except ValueError:
                            pass
                cards = "".join([
                    f"""<div class="coral-card">
                          <div class="coral-label">{lbl}</div>
                          <div class="coral-val">{v:.2f}</div>
                          <div class="coral-bar"><div class="coral-fill" style="width:{min(int(v*100),100)}%"></div></div>
                        </div>"""
                    for lbl, v in thresholds
                ])
                st.markdown(f'<div class="section-label">CORAL Thresholds</div><div class="coral-grid">{cards}</div>', unsafe_allow_html=True)

            # Top-3
            rows = "".join([
                f"""<div class="top3-row">
                      <span>{name}</span>
                      <div style="display:flex;align-items:center;gap:8px;">
                        <div class="top3-bar"><div class="top3-fill" style="width:{int(prob*100)}%"></div></div>
                        <span class="top3-pct">{prob:.1%}</span>
                      </div>
                    </div>"""
                for name, prob in result["top3_food"]
            ])
            st.markdown(f'<div class="section-label">Top-3 Food Recognition</div>{rows}', unsafe_allow_html=True)
        else:
            st.markdown('<div class="idle-msg">Upload or select a plate photo to see predictions</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — OPTIMIZATION
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "opt":

    left, right = st.columns([1.1, 1.9])

    # ── LEFT CARD ────────────────────────────────────────────────────────
    with left:
        st.markdown("""
        <div class="cs-card">
          <div class="cs-card-title"><span class="dot-blue"></span> Plate Photo Upload</div>
          <p style="font-size:0.87em;color:#777;margin:0 0 14px;line-height:1.5;">
            Upload a leftover plate image to update demand and consumption ratios,
            then recalculate optimized serving sizes.
          </p>
        </div>
        """, unsafe_allow_html=True)

        opt_img_path = None
        uploaded_opt = st.file_uploader(
            "Drop a plate photo here",
            type=["jpg", "jpeg", "png"],
            key="opt_upload",
            label_visibility="collapsed",
        )
        if uploaded_opt:
            b64 = uploaded_to_b64(uploaded_opt)
            st.markdown(f'<img src="{b64}" class="preview-img">', unsafe_allow_html=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                uploaded_opt.seek(0)
                tmp.write(uploaded_opt.read())
                opt_img_path = tmp.name

        update_btn = st.button(
            "Update Demand & Recalculate",
            key="update_opt",
            disabled=(opt_img_path is None),
            use_container_width=True,
        )

        if update_btn and opt_img_path:
            with st.spinner("Analysing plate…"):
                try:
                    model  = get_model()
                    result = m5_predict(model, "M5", opt_img_path)
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                    st.stop()

            top3 = result.get("top3_food", [])
            if top3:
                new_ratio   = result["pred_r"]
                matched_key = find_dish_key(top3[0][0], st.session_state.dishes)
                if matched_key:
                    st.session_state.dishes[matched_key]["R_i"]    = new_ratio
                    st.session_state.dishes[matched_key]["D_base"] += 1
                    st.markdown(f'<div class="status-ok">✓ Updated <b>{matched_key}</b> — ratio set to {new_ratio*100:.0f}%</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-warn">⚠ No dish matched "{top3[0][0]}"</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
            if st.button("🔄 Reset to defaults", key="reset_opt", use_container_width=True):
                st.session_state.dishes = copy.deepcopy(DISHES_DATA)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT CARD ───────────────────────────────────────────────────────
    with right:
        optimized = run_serving_optimization(st.session_state.dishes)

        rows_html = ""
        labels_list, std_list, opt_list = [], [], []
        for name, d in st.session_state.dishes.items():
            ov = optimized[name]
            rows_html += f"""
            <tr>
              <td>{name}</td>
              <td>{d['P']:,}</td>
              <td>{d['C']}</td>
              <td>{d['D_base']}</td>
              <td>{int(d['R_i']*100)}%</td>
              <td>{d['S_base']}g</td>
              <td class="optimized-val">{ov:.2f}g</td>
            </tr>"""
            labels_list.append(name)
            std_list.append(d["S_base"])
            opt_list.append(round(ov, 2))

        st.markdown(f"""
        <div class="cs-card">
          <div class="cs-card-title"><span class="dot-green"></span> Optimized Serving Sizes</div>
          <div style="overflow-x:auto;">
            <table class="opt-table">
              <thead><tr>
                <th>Dish</th><th>Price (IDR)</th><th>Cost / g</th>
                <th>Demand (Q)</th><th>Cons. Ratio</th>
                <th>Std Size</th><th>Optimized Size</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Chart.js charts — embedded, matching optimization.html exactly
        labels_js    = json.dumps(labels_list)
        std_js       = json.dumps(std_list)
        opt_js       = json.dumps(opt_list)
        reduction_js = json.dumps([
            round((o - s) / s * 100, 2) if s else 0
            for s, o in zip(std_list, opt_list)
        ])

        components.html(f"""
        <!DOCTYPE html>
        <html>
        <head>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
          body {{ margin:0; background:transparent; font-family:system-ui,-apple-system,sans-serif; }}
          .chart-card {{
            background:#fff; border-radius:12px; padding:18px;
            border:0.5px solid #e0ddd6; margin-bottom:16px;
          }}
          .chart-title {{
            font-size:11px; font-weight:600; letter-spacing:.07em;
            text-transform:uppercase; color:#5f5e5a; margin-bottom:14px;
            display:flex; align-items:center; gap:7px;
          }}
          .dot {{ width:7px;height:7px;border-radius:50%;background:#1d9e75;display:inline-block; }}
        </style>
        </head>
        <body>
        <div class="chart-card">
          <div class="chart-title"><span class="dot"></span> Std Size vs Optimized Size</div>
          <canvas id="sizeChart" height="110"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-title"><span class="dot"></span> Serving Size Reduction (%)</div>
          <canvas id="redChart" height="110"></canvas>
        </div>

        <script>
        new Chart(document.getElementById('sizeChart').getContext('2d'), {{
          type: 'bar',
          data: {{
            labels: {labels_js},
            datasets: [
              {{ label:'Standard Size (g)',  data:{std_js}, backgroundColor:'#d7ccc8', borderColor:'#8d6e63', borderWidth:1 }},
              {{ label:'Optimized Size (g)', data:{opt_js}, backgroundColor:'#6d4c41', borderColor:'#4e342e', borderWidth:1 }}
            ]
          }},
          options: {{
            responsive:true,
            plugins:{{ legend:{{ labels:{{ color:'#4e342e', font:{{size:12}} }} }} }},
            scales:{{
              x:{{ ticks:{{color:'#5d4037',maxRotation:35}}, grid:{{color:'#efebe9'}} }},
              y:{{ beginAtZero:true, ticks:{{color:'#5d4037'}}, grid:{{color:'#efebe9'}},
                   title:{{display:true,text:'Serving Size (g)',color:'#5d4037'}} }}
            }}
          }}
        }});

        const red = {reduction_js};
        new Chart(document.getElementById('redChart').getContext('2d'), {{
          type: 'bar',
          data: {{
            labels: {labels_js},
            datasets: [{{
              label: 'Reduction (%)',
              data: red,
              backgroundColor: red.map(v => v < 0 ? '#a1887f' : '#bcaaa4'),
              borderColor: '#5d4037',
              borderWidth: 1
            }}]
          }},
          options: {{
            responsive:true,
            plugins:{{
              legend:{{ labels:{{color:'#4e342e'}} }},
              tooltip:{{ callbacks:{{ label: ctx => ctx.raw.toFixed(2)+'%' }} }}
            }},
            scales:{{
              x:{{ ticks:{{color:'#5d4037',maxRotation:35}}, grid:{{color:'#efebe9'}} }},
              y:{{ ticks:{{color:'#5d4037', callback: v => v+'%'}}, grid:{{color:'#efebe9'}},
                   title:{{display:true,text:'Reduction (%)',color:'#5d4037'}} }}
            }}
          }}
        }});
        </script>
        </body>
        </html>
        """, height=700, scrolling=False)