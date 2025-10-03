# =========================
# CAD Prediction App (updated for USA + Japan models)
# =========================
import os
import json
import pickle
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional libs
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

# Optional: ReportLab for PDF
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors as RLcolors
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

# ML
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score  # used in PDF table sometimes
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Ensure XGBoost is available
try:
    import xgboost  # noqa: F401
except ImportError:
    st.error("❌ XGBoost is not installed. Add `xgboost` to requirements.txt")
    st.stop()

# ---------------------------
# Page config + minimal CSS
# ---------------------------
st.set_page_config(page_title="CAD Prediction App", page_icon="🫀", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button {
            background-color: #007bff; color: white;
            border-radius: 8px; padding: 0.6em 1.2em; font-weight: bold;
        }
        .stButton>button:hover { background-color: #0056b3; color: white; }
        h1, h2, h3 { color: #2c3e50; }
        .reportview-container .markdown-text-container { font-size: 1.1em; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
def scan_models():
    """
    Find best_model_*.pkl and pair with adjacent JSON, normalizing 'use' -> 'usa'.
    Returns an ordered dict-like: key -> {'pkl': ..., 'json': ...}
    """
    candidates = {}
    for f in os.listdir("."):
        if f.startswith("best_model_") and f.endswith(".pkl"):
            tag = f[len("best_model_"):-4].lower().strip()
            if tag == "use":  # normalize typo/variant
                tag_norm = "usa"
            else:
                tag_norm = tag
            j = f"best_model_{tag}.json"
            if not os.path.exists(j):
                # try normalized json name too
                j_alt = f"best_model_{tag_norm}.json"
                j = j if os.path.exists(j) else (j_alt if os.path.exists(j_alt) else None)
            candidates[tag_norm] = {"pkl": f, "json": j}
    # keep a stable order: USA first if present
    ordered = {}
    for key in ["usa", "japan"]:
        if key in candidates:
            ordered[key] = candidates[key]
    for k, v in candidates.items():
        if k not in ordered:
            ordered[k] = v
    return ordered

def load_model(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def find_column_transformer(estimator):
    """Return the first ColumnTransformer found inside a Pipeline (or None)."""
    if isinstance(estimator, ColumnTransformer):
        return estimator
    if isinstance(estimator, Pipeline):
        for name, step in estimator.named_steps.items():
            found = find_column_transformer(step)
            if found is not None:
                return found
    return None

def get_base_columns_from_ct(ct: ColumnTransformer):
    """
    Pull the original column names the CT expects (union of all transformer column lists).
    Handles common cases where cols are name-lists.
    """
    base_cols = []
    try:
        for tr_name, tr, cols in ct.transformers_:
            if tr == "drop":
                continue
            if isinstance(cols, list):
                base_cols.extend([str(c) for c in cols])
            # if slice or array-like: try to resolve via feature_names_in_ when present
            elif hasattr(ct, "feature_names_in_"):
                try:
                    if isinstance(cols, slice):
                        sel = ct.feature_names_in_[cols]
                    else:
                        sel = ct.feature_names_in_[cols]
                    base_cols.extend([str(c) for c in sel])
                except Exception:
                    pass
    except Exception:
        pass
    # de-dup preserving order
    seen = set(); out = []
    for c in base_cols:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def align_to_expected_columns(df: pd.DataFrame, expected_cols):
    """
    Ensure df has all expected columns. Add missing columns with NaN.
    Extra columns are harmless; Pipeline will pick what it needs.
    """
    X = df.copy()
    for col in expected_cols:
        if col not in X.columns:
            X[col] = np.nan
    return X

def normalize_labels_from_df(df: pd.DataFrame):
    """
    Try to produce y labels as {'cad','control'} from either 'Status' (cad/control)
    or 'CARDIOVASCULAR_DISEASE' (True/False or 1/0). Returns (y_series, target_col_used).
    """
    if "Status" in df.columns:
        y = df["Status"].astype(str).str.lower().map({"cad": "cad", "control": "control"})
        return y, "Status"
    if "CARDIOVASCULAR_DISEASE" in df.columns:
        raw = df["CARDIOVASCULAR_DISEASE"].astype(str).str.lower().str.strip()
        pos = {"1", "true", "yes", "y", "cad"}
        y = raw.apply(lambda v: "cad" if v in pos else "control")
        return y, "CARDIOVASCULAR_DISEASE"
    return None, None

def load_metrics_json(json_path):
    """Read metrics/params JSON if available; returns dict or None."""
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception:
        return None

# ---------------------------
# Sidebar & Navigation
# ---------------------------
if os.path.exists("Main.jpg"):
    st.sidebar.image("Main.jpg", caption="Gut Microbiome & CAD", use_container_width=True)

st.sidebar.title("🫀 CAD Prediction App")

page = st.sidebar.radio("Navigate to:", ["Home", "CAD Prediction Tool", "Datasets", "About"])

# ---------------------------
# Page: Home
# ---------------------------
if page == "Home":
    st.markdown("<h1 style='text-align:center;'>🫀 CAD Prediction using Gut Microbiome</h1>", unsafe_allow_html=True)
    st.write("""
    <div style="text-align: center; font-size: 18px;">
        Coronary Artery Disease (CAD) is a leading cause of death worldwide.<br>
        This project explores how <b>gut microbiome composition</b> can help predict CAD risk using machine learning.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("""
        ### Why this project?
        - CAD is a critical health concern globally.
        - Gut microbiome research provides new insights into disease prediction.
        - We combine **bioinformatics + AI** to build a CAD prediction pipeline.

        📊 This app lets you:
        - Upload your dataset
        - Run predictions
        - Generate performance reports
        """)
    with c2:
        if os.path.exists("Main.jpg"):
            st.image("Main.jpg", caption="Gut Microbiome & CAD", use_container_width=True)

# ---------------------------
# Page: CAD Prediction Tool
# ---------------------------
elif page == "CAD Prediction Tool":
    st.markdown("## 🔬 CAD Prediction Tool")

    available = scan_models()
    if not available:
        st.warning("No `best_model_*.pkl` files found. Place your models in the app folder and reload.")
        st.stop()

    # Selector
    labels = {"usa": "USA model", "japan": "Japan model"}
    show_options = [labels.get(k, k.upper()) for k in available.keys()]
    choice = st.selectbox("Select model", options=show_options, index=0)
    choice_key = [k for k, lbl in labels.items() if lbl == choice]
    selected_key = choice_key[0] if choice_key else list(available.keys())[0]
    model_files = available[selected_key]

    # Show saved metrics/params if JSON present
    if model_files.get("json"):
        info = load_metrics_json(model_files["json"])
        if info:
            with st.expander("📈 Model snapshot (from JSON)"):
                st.json(info)

    uploaded_file = st.file_uploader("Upload microbiome dataset (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview")
            st.dataframe(data.head())

            # Build y if possible
            y_true, target_col = normalize_labels_from_df(data)
            if y_true is None:
                st.warning("Target not found. Add a 'Status' (cad/control) or 'CARDIOVASCULAR_DISEASE' column to compute metrics.")
            # Features (raw) – let the pipeline handle preprocessing
            X_raw = data.drop(columns=[c for c in ["Status", "CARDIOVASCULAR_DISEASE"] if c in data.columns], errors="ignore")

            # Load model
            try:
                model_obj = load_model(model_files["pkl"])
            except Exception as err:
                st.error(f"Couldn’t load model: {model_files['pkl']}\n\nError: {err}")
                st.stop()

            # Align columns to what the fitted ColumnTransformer expects
            ct = find_column_transformer(model_obj)
            if ct is not None:
                expected_cols = get_base_columns_from_ct(ct)
                X_in = align_to_expected_columns(X_raw, expected_cols)
                # Keep only expected columns to avoid accidental dtype/promotions from extras
                X_in = X_in[[c for c in expected_cols if c in X_in.columns]]
            else:
                # If model isn't a pipeline, just pass the raw frame
                X_in = X_raw

            if st.button("🚀 Run CAD Prediction"):
                preds = model_obj.predict(X_in)

                # Try to map numeric classes to strings; if we can't, show ints
                label_map = {0: "cad", 1: "control"}
                pred_labels = [label_map.get(int(p), str(p)) for p in preds]
                data["Predicted_Status"] = pred_labels

                if y_true is not None:
                    data["Actual_Status"] = y_true

                st.success("✅ Predictions completed!")
                st.dataframe(data.head())

                # Prediction distribution
                pred_counts = pd.Series(pred_labels).value_counts()
                fig1, ax1 = plt.subplots()
                if HAS_SEABORN:
                    sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax1)
                else:
                    ax1.bar(pred_counts.index, pred_counts.values)
                ax1.set_title("Predicted classes")
                ax1.set_xlabel("Label"); ax1.set_ylabel("Count")
                st.pyplot(fig1)

                # Confusion matrix + accuracy (only if ground truth present)
                fig2 = None; acc = None
                if y_true is not None:
                    cm = confusion_matrix(y_true, pred_labels, labels=["cad", "control"])
                    fig2, ax2 = plt.subplots()
                    if HAS_SEABORN:
                        sns.heatmap(cm, annot=True, fmt="d",
                                    xticklabels=["cad", "control"],
                                    yticklabels=["cad", "control"],
                                    ax=ax2)
                    else:
                        im = ax2.imshow(cm, aspect="auto")
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                ax2.text(j, i, str(cm[i, j]), ha="center", va="center")
                        ax2.set_xticks([0,1]); ax2.set_xticklabels(["cad", "control"])
                        ax2.set_yticks([0,1]); ax2.set_yticklabels(["cad", "control"])
                        ax2.figure.colorbar(im, ax=ax2)
                    ax2.set_title("Confusion Matrix")
                    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
                    st.pyplot(fig2)

                    acc = accuracy_score(y_true, pred_labels)
                    st.markdown(f"### 🎯 Model Accuracy (on your file): **{acc:.2f}**")

                # Cache for PDF
                st.session_state["data"] = data.copy()
                st.session_state["fig1"] = fig1
                st.session_state["fig2"] = fig2
                st.session_state["acc"] = acc
                st.session_state["model_key"] = selected_key
                st.session_state["model_json"] = model_files.get("json")

            # PDF Report
            if st.button("📄 Generate PDF Report"):
                if "data" not in st.session_state:
                    st.error("Run predictions first to enable report download.")
                elif not HAS_REPORTLAB:
                    st.info("PDF export requires ReportLab. Add `reportlab` to requirements.txt to enable this.")
                else:
                    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    pdf_file = tmp_pdf.name
                    tmp_pdf.close()

                    doc = SimpleDocTemplate(pdf_file)
                    styles = getSampleStyleSheet()
                    story = []
                    story.append(Paragraph("CAD Prediction Report", styles["Title"]))
                    story.append(Spacer(1, 20))

                    # Include saved snapshot metrics if available
                    if st.session_state.get("model_json"):
                        meta = load_metrics_json(st.session_state["model_json"])
                        if meta:
                            story.append(Paragraph("Model snapshot (from training JSON):", styles["Heading2"]))
                            story.append(Paragraph(json.dumps(meta, indent=2), styles["Code"]))
                            story.append(Spacer(1, 12))

                    if st.session_state.get("acc") is not None:
                        story.append(Paragraph(f"Accuracy (your file): {st.session_state['acc']:.2f}", styles["Normal"]))
                        story.append(Spacer(1, 10))

                    # Classification report (only if y present)
                    if "Actual_Status" in st.session_state["data"].columns:
                        rep = classification_report(
                            st.session_state["data"]["Actual_Status"],
                            st.session_state["data"]["Predicted_Status"],
                            output_dict=True,
                            zero_division=0
                        )
                        rep_df = pd.DataFrame(rep).transpose().round(2)
                        table_data = [["Class", "Precision", "Recall", "F1-Score", "Support"]]
                        for cls in rep_df.index:
                            if cls == "accuracy": 
                                continue
                            row = [
                                cls,
                                rep_df.loc[cls].get("precision", ""),
                                rep_df.loc[cls].get("recall", ""),
                                rep_df.loc[cls].get("f1-score", ""),
                                int(rep_df.loc[cls].get("support", 0)),
                            ]
                            table_data.append(row)
                        tbl = Table(table_data, hAlign="LEFT")
                        tbl.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), RLcolors.grey),
                            ('TEXTCOLOR',(0,0),(-1,0),RLcolors.whitesmoke),
                            ('ALIGN',(1,1),(-1,-1),'CENTER'),
                            ('GRID', (0,0), (-1,-1), 1, RLcolors.black),
                            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ]))
                        story.append(Paragraph("Classification Report", styles["Heading2"]))
                        story.append(tbl)
                        story.append(Spacer(1, 20))

                    # Figures
                    for fig, title in [(st.session_state.get("fig1"), "Prediction Distribution"),
                                       (st.session_state.get("fig2"), "Confusion Matrix")]:
                        if fig is not None:
                            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            fig.savefig(tmp_img.name, bbox_inches="tight", facecolor="white")
                            story.append(Paragraph(title, styles["Heading2"]))
                            story.append(RLImage(tmp_img.name, width=420, height=260))
                            story.append(Spacer(1, 16))
                            tmp_img.close()

                    doc.build(story)
                    with open(pdf_file, "rb") as f:
                        st.download_button(
                            label="📥 Download PDF",
                            data=f.read(),
                            file_name=f"CAD_Prediction_Report_{st.session_state.get('model_key','model')}.pdf",
                            mime="application/pdf"
                        )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ---------------------------
# Page: Datasets
# ---------------------------
elif page == "Datasets":
    st.markdown("## 📂 Sample Datasets")
    st.write("Use these to try the app, or upload your own in the tool.")

    sample_files = {
        "🧬 American Gut Microbiome (USA)": "dataset_filtered.csv",
        "🍣 Japan Gut Microbiome": "PRJDB6472.csv",
    }
    for name, file in sample_files.items():
        if os.path.exists(file):
            with open(file, "rb") as f:
                st.download_button(label=f"⬇️ Download {name}", data=f, file_name=file, mime="text/csv")
        else:
            st.warning(f"{file} not found in the app folder.")

# ---------------------------
# Page: About
# ---------------------------
elif page == "About":
    st.markdown("## ℹ️ About this Project")
    st.write("""
    This project studies the link between gut microbiome and Coronary Artery Disease (CAD).
    Models are trained per-region and packaged as scikit-learn pipelines with preprocessing.
    """)

    # Summarize available model snapshots (metrics/params) if JSON files exist
    st.markdown("### Model snapshots (from JSON)")
    rows = []
    for key, files in scan_models().items():
        meta = load_metrics_json(files.get("json")) if files.get("json") else None
        if meta:
            rows.append({
                "Model": key.upper(),
                "Metrics": meta.get("metrics"),
                "Best Params": meta.get("best_params"),
            })
    if rows:
        st.json(rows)
    else:
        st.info("No metrics JSON found next to your .pkl files.")

    st.markdown("""
    **Tech stack**
    - XGBoost, scikit-learn pipelines
    - Streamlit for the UI
    - ReportLab for PDF export *(optional)*
    """)

# ---------------------------
# Extras
# ---------------------------
with st.expander("🔎 Full directory listing"):
    st.write("\n".join(sorted(os.listdir("."))))
