import os
import sys
import tempfile
from Serving_Size_Optimization.optimization_engine import run_serving_optimization

DISHES_DATA = {
    'Orek Tempe':       {'P': 4000,  'C': 15, 'S_base': 29.59,  'D_base': 22, 'R_i': 0.61, 'alpha': 0.10},
    'Bali Tahu':   {'P': 4000,  'C': 12, 'S_base': 34.64,  'D_base': 22, 'R_i': 0.47, 'alpha': 0.8},
    'Telur Mata Sapi':  {'P': 6000,  'C': 45, 'S_base': 47.75,  'D_base': 20, 'R_i': 0.75, 'alpha': 0.8},
    'Ayam bumbu bistik': {'P': 15000, 'C': 75, 'S_base': 55.32,  'D_base': 22, 'R_i': 0.67, 'alpha': 1.6},
    'Ayam laos':   {'P': 15000, 'C': 85, 'S_base': 52.25,  'D_base': 16, 'R_i': 0.30, 'alpha': 1.6},
    'Rendang':    {'P': 18000, 'C': 95, 'S_base': 54.39,  'D_base': 18, 'R_i': 0.54, 'alpha': 1.6},
    'Bali Telur':   {'P': 7000,  'C': 48, 'S_base': 56.10,  'D_base': 21, 'R_i': 0.73, 'alpha': 0.8},
    'Ikan Acar Kuning':        {'P': 16000, 'C': 80, 'S_base': 39.00,  'D_base': 20, 'R_i': 0.53, 'alpha': 1.6},
    'Nasi':        {'P': 5000,  'C': 6,  'S_base': 146.32, 'D_base': 78, 'R_i': 0.62, 'alpha': 0.10},
}

MODEL_HANDOFF = os.path.join(os.path.dirname(__file__), "Model_Handoff")
sys.path.insert(0, MODEL_HANDOFF)

import dataset
dataset.EXCEL_PATH = r"/Users/cynthiaathena/Documents/S2/School/Spring/242B/Project/food detection/Model_Handoff/data_original.xlsx"

from flask import Flask, render_template, request, jsonify, redirect, url_for
from predict import load_model, predict as m5_predict

app = Flask(__name__)

CKPT = os.path.join(os.path.dirname(__file__), "Model_Handoff", "ckpt_M5.pt")
model = load_model("M5", CKPT)

PRESET_MENUS = [1, 7, 9, 11, 12, 13, 15, 23, 31]


@app.route("/")
def index():
    return render_template("m5_prediction.html", menus=PRESET_MENUS)


@app.route("/api/presets")
def get_presets():
    return jsonify([
        {
            "name": f"menu{n}",
            "after_url":  f"/static/presets/menu{n}.jpg",
            "before_url": f"/static/presets/menu{n}_before.jpg",
        }
        for n in PRESET_MENUS
    ])


@app.route("/api/predict", methods=["POST"])
def run_predict():
    try:
        if "image" in request.files:
            file = request.files["image"]
            suffix = os.path.splitext(file.filename)[-1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file.save(tmp.name)
                img_path = tmp.name
        else:
            preset = request.form.get("preset")
            img_path = os.path.join(
                os.path.dirname(__file__), "static", "presets", f"{preset}.jpg"
            )

        result = m5_predict(model, "M5", img_path)

        thresholds = []
        if result["threshold_str"]:
            for part in result["threshold_str"].split("  "):
                label, val = part.split("=")
                thresholds.append({"label": label, "value": float(val)})

        return jsonify({
            "pred_r":    result["pred_r"],
            "bin":       result["bin"],
            "top3_food": [{"name": n, "prob": round(p, 3)}
                          for n, p in result["top3_food"]],
            "thresholds": thresholds,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── NEW: JSON endpoint used by the redesigned optimization page ──────────────
@app.route("/optimization/predict_json", methods=["GET","POST"])
def opt_predict_json():
    """
    Accepts a multipart image upload, runs the M5 model, updates DISHES_DATA,
    re-runs the optimiser, and returns the full updated table as JSON.
    """
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        img_path = tmp.name

    try:
        result = m5_predict(model, "M5", img_path)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {e}"}), 500

    # ── BUG FIX ──────────────────────────────────────────────────────────────
    # result["top3_food"] is a list of (name, prob) tuples, e.g.:
    #   [('11_Rendang', 0.95), ('1_Nasi', 0.02), ('9_BaliTelur', 0.01)]
    # We need the *string* name from the first tuple — not the list itself.
    # Using the list as a dict key caused: TypeError: unhashable type: 'list'
    top3 = result.get("top3_food", [])
    if not top3:
        return jsonify({"error": "Model returned no food predictions"}), 500

    top_food_name = top3[0][0]   # raw string from model, e.g. '11_Rendang'
    new_ratio     = result["pred_r"]

    # ── Robust key lookup ────────────────────────────────────────────────────
    # The model label may differ in case or separators from DISHES_DATA keys.
    # Strategy: exact match first, then case-insensitive, then numeric-prefix.
    def find_dish_key(raw_name, dishes):
        # 1. Exact match
        if raw_name in dishes:
            return raw_name
        # 2. Case-insensitive match
        raw_lower = raw_name.lower()
        for key in dishes:
            if key.lower() == raw_lower:
                return key
        # 3. Match by numeric prefix (e.g. model returns "11" → "11_Rendang")
        raw_prefix = raw_name.split("_")[0]
        for key in dishes:
            if key.split("_")[0] == raw_prefix:
                return key
        # 4. Substring match — model label contained in key or vice versa
        for key in dishes:
            if raw_lower in key.lower() or key.lower() in raw_lower:
                return key
        return None

    matched_key = find_dish_key(top_food_name, DISHES_DATA)
    updated_dish = None
    if matched_key:
        DISHES_DATA[matched_key]["R_i"]    = new_ratio
        DISHES_DATA[matched_key]["D_base"] += 1
        updated_dish = matched_key

    # Re-run optimisation with updated data
    optimized = run_serving_optimization(DISHES_DATA)

    dishes_payload = [
        {
            "name":      name,
            "P":        d["P"],
            "C":    d["C"],
            "D_base":    d["D_base"],
            "R_i":       d["R_i"],
            "S_base":    d["S_base"],
            "optimized": optimized[name],
        }
        for name, d in DISHES_DATA.items()
    ]

    return jsonify({
        "dishes":       dishes_payload,
        "updated_dish": updated_dish,
        "new_ratio":    new_ratio,
        "raw_prediction": top_food_name,
    })


# ── Keep the old form-POST endpoint as a redirect fallback (optional) ────────
@app.route("/optimization/predict", methods=["POST"])
def opt_predict():
    """Legacy form-POST — now just delegates to the JSON endpoint logic."""
    return redirect(url_for("optimization"))


@app.route("/optimization")
def optimization():
    optimized_results = run_serving_optimization(DISHES_DATA)
    return render_template(
        "optimization.html",
        dishes=DISHES_DATA,
        optimized=optimized_results,
    )


if __name__ == "__main__":
    app.run(debug=True)