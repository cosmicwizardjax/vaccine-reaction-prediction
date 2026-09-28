import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# ============================================================
# SETTINGS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PKL_FILE = BASE_DIR / "reaction_model.pkl"
MODEL_CBM_FILE = BASE_DIR / "reaction_model.cbm"
METADATA_FILE = BASE_DIR / "model_metadata.pkl"
DATA_FILE = BASE_DIR / "data" / "final ugrc (1).xlsx"
SHEET_NAME = "compiled data all"

OTHER_LABEL = "OTHER REPORTED REACTION (PRESENT IN DATASET)"
TOP_N_REACTIONS = 3
GRADE_COLUMN = "CLASSIFICATION* BY NATIONAL AEFI COMMITTEE"


# ============================================================
# PAGE STYLE
# ============================================================
st.set_page_config(
    page_title="Vaccine Reaction Prediction",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(110deg, #0d2740, #167b8c);
        color: white;
        padding: 28px 34px;
        border-radius: 0 0 30px 30px;
        margin-bottom: 16px;
    }
    .hero h1 { margin: 0; font-size: 2.35rem; }
    .hero p { margin: 14px 0 0; font-size: 1rem; opacity: .92; }
    .notice {
        background: #e8f2ff;
        color: #07549a;
        padding: 16px 18px;
        border-radius: 9px;
        margin: 10px 0 18px;
    }
    .section-title { margin-top: 8px; }
    .reaction-card {
        background: #f8fafc;
        border: 1px solid #d9e2ec;
        border-radius: 14px;
        padding: 18px 20px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }
    .reaction-title { font-size: 1.18rem; font-weight: 700; margin-bottom: 14px; }
    .reaction-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 12px; }
    .reaction-grid > div { background: white; border-radius: 10px; padding: 10px 12px; border: 1px solid #e5e7eb; }
    .reaction-grid .label { display: block; font-size: 0.78rem; color: #64748b; margin-bottom: 4px; }
    .reaction-grid strong { font-size: 1rem; }
    .reaction-text { margin-top: 9px; line-height: 1.5; }
    @media (max-width: 800px) { .reaction-grid { grid-template-columns: 1fr; } }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>💉 Vaccine Reaction Prediction</h1>
      <p>Dataset-based prediction • interactive analytics • personalized safety guidance • offline-ready</p>
    </div>
    <div class="notice">
      This application provides a dataset-based prediction category. It is not a diagnosis and does not prove that a vaccine caused an event. Do not use it as a substitute for professional medical care.
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_model_and_metadata():
    metadata = joblib.load(METADATA_FILE)

    # Prefer the model file that is actually present in the repository.
    # This makes the app work on Streamlit Cloud even when the optional
    # CatBoost .cbm file is not uploaded.
    if MODEL_PKL_FILE.exists():
        model = joblib.load(MODEL_PKL_FILE)
        return model, metadata

    if MODEL_CBM_FILE.exists():
        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost is required to load reaction_model.cbm. "
                "Install it using: python -m pip install catboost"
            )
        model = CatBoostClassifier()
        model.load_model(str(MODEL_CBM_FILE))
        return model, metadata

    raise FileNotFoundError(
        "No trained model file was found. Expected reaction_model.pkl "
        "or reaction_model.cbm in the repository root."
    )

try:
    model, metadata = load_model_and_metadata()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()

# ============================================================
# DATASET FOR ANALYTICS
# ============================================================
@st.cache_data
def load_dataset():
    if not DATA_FILE.exists():
        return None
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df

dataset = load_dataset()

# ============================================================
# HELPERS
# ============================================================
def clean_series(series):
    return (
        series.astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.upper()
    )


def get_age_group(age):
    if age < 18:
        return "UNDER 18"
    if age <= 30:
        return "18-30"
    if age <= 44:
        return "31-44"
    if age <= 59:
        return "45-59"
    return "60+"


def create_input(age, sex, vaccine):
    age_group = get_age_group(age)
    return pd.DataFrame([{
        "AGE (IN YEARS)": float(age),
        "SEX": str(sex),
        "VACCINE": str(vaccine),
        "AGE_GROUP": str(age_group),
        "VACCINE_AGE_GROUP": f"{vaccine}_{age_group}",
        "VACCINE_SEX": f"{vaccine}_{sex}",
    }])


def get_risk_level(probability):
    p = probability * 100
    if p >= 50:
        return "HIGH"
    if p >= 20:
        return "MODERATE"
    return "LOW"


def format_reaction_name(reaction):
    reaction = str(reaction).strip()
    if reaction == OTHER_LABEL:
        return "OTHER REPORTED REACTION"
    return reaction


def get_top_reactions(model, X, top_n=TOP_N_REACTIONS):
    probabilities = np.asarray(model.predict_proba(X)[0], dtype=float)
    classes = np.asarray(model.classes_)
    sorted_indices = np.argsort(probabilities)[::-1]
    top_indices = sorted_indices[:min(top_n, len(sorted_indices))]
    rows = []
    for rank, index in enumerate(top_indices, start=1):
        probability = float(probabilities[index])
        rows.append({
            "Rank": rank,
            "Reaction": format_reaction_name(classes[index]),
            "Probability": probability,
            "Probability (%)": probability * 100,
            "Risk Level": get_risk_level(probability),
        })
    return pd.DataFrame(rows)


def get_historical_grade(reaction):
    """Return the AEFI classification recorded for the predicted diagnosis.

    The value is derived only from the supplied Excel dataset. If more than one
    classification was recorded for the same diagnosis, the most frequently
    recorded classification is shown and the other recorded classifications are
    mentioned so that the app does not silently invent a single grade.
    """
    if dataset is None or GRADE_COLUMN not in dataset.columns or "DIAGNOSIS" not in dataset.columns:
        return "Not available"

    d = dataset.copy()
    d["DIAGNOSIS_CLEAN"] = clean_series(d["DIAGNOSIS"])
    d["GRADE_CLEAN"] = clean_series(d[GRADE_COLUMN])

    reaction_clean = str(reaction).strip().upper()
    if reaction_clean == OTHER_LABEL:
        counts = d["DIAGNOSIS_CLEAN"].value_counts()
        threshold = int(metadata.get("threshold", 25))
        common = set(counts[counts >= threshold].index)
        d = d[~d["DIAGNOSIS_CLEAN"].isin(common)]
    else:
        d = d[d["DIAGNOSIS_CLEAN"] == reaction_clean]

    d = d[d["GRADE_CLEAN"].notna() & (d["GRADE_CLEAN"] != "")]
    if d.empty:
        return "Not available"

    counts = d["GRADE_CLEAN"].value_counts()
    primary = str(counts.index[0])
    others = [str(x) for x in counts.index[1:]]
    if others:
        return f"{primary} (most common; {', '.join(others)} also recorded)"
    return primary


def get_historical_severity(reaction):
    """Summarize the highest level of reported outcome for this reaction, without counts."""
    if dataset is None or "REASON FOR REPORTING/ OUTCOME" not in dataset.columns or "DIAGNOSIS" not in dataset.columns:
        return "Not available", "No historical outcome information is available in the dataset."

    d = dataset.copy()
    d["DIAGNOSIS_CLEAN"] = clean_series(d["DIAGNOSIS"])
    d["OUTCOME_CLEAN"] = clean_series(d["REASON FOR REPORTING/ OUTCOME"])
    reaction_clean = str(reaction).strip().upper()

    if reaction_clean == OTHER_LABEL:
        counts = d["DIAGNOSIS_CLEAN"].value_counts()
        threshold = int(metadata.get("threshold", 25))
        common = set(counts[counts >= threshold].index)
        d = d[~d["DIAGNOSIS_CLEAN"].isin(common)]
    else:
        d = d[d["DIAGNOSIS_CLEAN"] == reaction_clean]

    outcomes = set(d["OUTCOME_CLEAN"].dropna())
    if not outcomes:
        return "Not available", "No historical outcome information is available in the dataset."

    has_death = any("DEATH" in x for x in outcomes)
    has_hospital = any("HOSPITAL" in x or "HOSPITALI" in x for x in outcomes)
    has_severe = any("SEVERE" in x for x in outcomes)

    if has_death:
        return "Very High", "Hospitalization, severe outcomes and death have been reported for this reaction."
    if has_hospital and has_severe:
        return "High", "Hospitalization and severe outcomes have been reported for this reaction."
    if has_hospital:
        return "Moderate–High", "Hospitalization has been reported for this reaction."
    if has_severe:
        return "Moderate–High", "Severe outcomes have been reported for this reaction."
    return "Lower", "No severe, hospitalized, or fatal outcome is represented in the available records for this reaction."


def get_reaction_guidance(reaction):
    key = str(reaction).strip().upper()
    guidance = {
        "ANAPHYLAXIS": "If there is breathing difficulty, throat/tongue swelling, fainting, or rapid worsening, seek emergency medical care immediately.",
        "THROMBOSIS WITH THROMBOCYTOPENIA SYNDROME": "Severe/persistent headache, abdominal or chest pain, leg swelling, shortness of breath, unusual bruising or bleeding require urgent medical assessment.",
        "SUDDEN CARDIAC DEATH": "This is a serious dataset category. Collapse, chest pain, severe breathlessness, or loss of consciousness requires emergency medical services.",
        "GUILLAIN BARRE SYNDROME": "Progressive weakness, difficulty walking, numbness/tingling, or breathing difficulty requires prompt medical assessment.",
        "FEVER": "Monitor temperature, rest and maintain fluids as tolerated. Seek medical advice if symptoms are severe, persistent, or worsening.",
        "ALLERGIC REACTION": "Monitor mild rash/itching and contact a healthcare professional for advice. Facial/throat swelling, breathing difficulty, fainting, or rapid worsening is an emergency.",
        "ANXIETY REACTION": "Provide reassurance and a calm environment. If symptoms are severe, persistent, or accompanied by chest pain, fainting, or breathing difficulty, seek medical assessment.",
        "COVID 19 DISEASE": "If compatible respiratory or other symptoms are present, contact a healthcare professional about testing and care.",
        "ERROR IN ADMINISTRATION": "Do not attempt to correct an administration error independently. Document it and seek assessment according to the applicable AEFI procedure.",
        "ACUTE FEBRILE REACTION": "Monitor fever and associated symptoms, maintain fluids as tolerated, and seek clinical assessment if symptoms are severe, persistent, or worsening.",
        "ACUTE GASTROENTERITIS": "Maintain fluids as tolerated and monitor for dehydration or worsening symptoms. Seek medical assessment for persistent vomiting/diarrhoea or other concerning symptoms.",
        "VASOVAGAL SYNCOPE": "If fainting or near-fainting occurs, ensure a safe position and seek medical assessment, especially for recurrent episodes, injury, chest pain, or breathing difficulty.",
    }
    return guidance.get(key, "The model identified this as a possible dataset category. Do not treat the prediction as a diagnosis. If symptoms are present, worsening, or concerning, seek assessment from a qualified healthcare professional.")


def get_general_precautions(top_reactions):
    return [
        "Monitor the person's actual symptoms and clinical condition; do not act on the prediction alone.",
        "Seek medical advice if symptoms are persistent, severe, or worsening.",
        "For breathing difficulty, loss of consciousness, severe swelling, or other rapidly worsening symptoms, seek emergency care immediately.",
    ]

# ============================================================
# TABS
# ============================================================
tab_predict, tab_analytics, tab_safety, tab_project = st.tabs([
    "🔎 Predict", "📊 Interactive Analytics", "🛡️ Safety & Resources", "ℹ️ Project"
])

# ============================================================
# PREDICT
# ============================================================
with tab_predict:
    st.markdown("## Enter patient & vaccination details")
    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        vaccine = st.selectbox("Vaccine", ["COVISHIELD", "COVAXIN", "CORBEVAX", "SPUTNIK V"])
    with c2:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45, step=1)
    with c3:
        sex = st.selectbox("Sex", ["MALE", "FEMALE"])

    predict_clicked = st.button("✨ Predict Possible Reactions", type="primary", use_container_width=True)

    if predict_clicked:
        X = create_input(age, sex, vaccine)
        try:
            top_reactions = get_top_reactions(model, X)
        except Exception as e:
            st.error("Error while generating predictions.")
            st.exception(e)
            st.stop()

        st.divider()
        st.subheader("📊 Top 3 Possible Predicted Reactions")
        st.caption("These are the three highest model-probability classes for the entered features. They are not a clinical diagnosis.")

        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        for _, row in top_reactions.iterrows():
            rank = int(row["Rank"])
            reaction = row["Reaction"]
            percentage = float(row["Probability (%)"])
            grade = get_historical_grade(reaction)
            severity, historical_outcome = get_historical_severity(reaction)

            st.markdown(f"""
            <div class="reaction-card">
                <div class="reaction-title">{medals.get(rank, '•')} #{rank} — {reaction}</div>
                <div class="reaction-grid">
                    <div><span class="label">Prediction</span><strong>{percentage:.2f}%</strong></div>
                    <div><span class="label">AEFI Classification</span><strong>{grade}</strong></div>
                    <div><span class="label">Potential seriousness</span><strong>{severity}</strong></div>
                </div>
                <div class="reaction-text"><b>Historical outcome:</b> {historical_outcome}</div>
                <div class="reaction-text"><b>Care:</b> {get_reaction_guidance(reaction)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("🛡️ General Precautions")
        for precaution in get_general_precautions(top_reactions):
            st.markdown(f"• {precaution}")

        st.caption("Historical outcome information is based on reported cases in the dataset and does not predict the outcome for an individual patient.")
        st.warning("⚠️ This application provides dataset-based decision support only. A prediction does not establish causality, diagnosis, or that an event will occur. For emergencies, seek emergency medical services immediately.")

# ============================================================
# INTERACTIVE ANALYTICS
# ============================================================
with tab_analytics:
    st.markdown("## 📊 Interactive Analytics")
    if dataset is None:
        st.warning(f"Analytics dataset was not found at: {DATA_FILE}")
        st.info("Keep the Excel file in the project's data folder as 'final ugrc (1).xlsx' for the analytics tab to populate.")
    else:
        df = dataset.copy()
        for col in ["SEX", "VACCINE", "REASON FOR REPORTING/ OUTCOME", "DIAGNOSIS", "CLASSIFICATION* BY NATIONAL AEFI COMMITTEE"]:
            if col in df.columns:
                df[col] = clean_series(df[col])
        if "AGE (IN YEARS)" in df.columns:
            df["AGE (IN YEARS)"] = pd.to_numeric(df["AGE (IN YEARS)"], errors="coerce")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Records", f"{len(df):,}")
        k2.metric("Vaccines", f"{df['VACCINE'].nunique():,}" if "VACCINE" in df.columns else "N/A")
        k3.metric("Age Groups", f"{df['AGE (IN YEARS)'].notna().sum():,}" if "AGE (IN YEARS)" in df.columns else "N/A")
        k4.metric("Reaction Categories", f"{df['DIAGNOSIS'].nunique():,}" if "DIAGNOSIS" in df.columns else "N/A")

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("### Vaccine Distribution")
            if "VACCINE" in df.columns:
                st.bar_chart(df["VACCINE"].value_counts().head(15))
        with a2:
            st.markdown("### Sex Distribution")
            if "SEX" in df.columns:
                st.bar_chart(df["SEX"].value_counts())

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("### Age Group Distribution")
            if "AGE (IN YEARS)" in df.columns:
                age_groups = pd.cut(
                    df["AGE (IN YEARS)"],
                    bins=[-np.inf, 17, 30, 44, 59, np.inf],
                    labels=["UNDER 18", "18-30", "31-44", "45-59", "60+"],
                ).value_counts().sort_index()
                st.bar_chart(age_groups)
        with b2:
            st.markdown("### Reporting / Outcome")
            if "REASON FOR REPORTING/ OUTCOME" in df.columns:
                st.bar_chart(df["REASON FOR REPORTING/ OUTCOME"].value_counts().head(15))

        st.markdown("### Top Reported Reaction Categories")
        if "DIAGNOSIS" in df.columns:
            st.bar_chart(df["DIAGNOSIS"].value_counts().head(15))

        if "CLASSIFICATION* BY NATIONAL AEFI COMMITTEE" in df.columns:
            st.markdown("### AEFI Classification")
            st.bar_chart(df["CLASSIFICATION* BY NATIONAL AEFI COMMITTEE"].value_counts().head(15))

        st.markdown("### Dataset Preview")
        # Convert preview values to text so mixed date/string columns render safely in Streamlit
        # (especially DATE OF VACCINATION (DD/MM/YYYY)).
        preview_df = df.head(25).copy()
        for col in preview_df.columns:
            preview_df[col] = preview_df[col].map(lambda x: x.strftime("%d/%m/%Y") if hasattr(x, "strftime") else ("" if pd.isna(x) else str(x)))
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

# ============================================================
# SAFETY & RESOURCES
# ============================================================
with tab_safety:
    st.markdown("## 🛡️ Safety & Resources")
    st.warning("The application is a dataset-based decision-support tool. It cannot diagnose a patient or establish that a vaccine caused an event.")

    st.markdown("### General precautions")
    for item in [
        "Use the prediction only as an informational aid and consider the person's actual symptoms and history.",
        "Do not delay emergency care because of a low model probability.",
        "For severe, rapidly worsening, or life-threatening symptoms, seek emergency medical services immediately.",
        "Keep a record of vaccine, timing, symptoms, treatment and outcome information for clinical discussion.",
    ]:
        st.markdown(f"• {item}")

    st.markdown("### Urgent warning signs")
    for item in [
        "Breathing difficulty or throat/tongue swelling",
        "Collapse, fainting or loss of consciousness",
        "Severe chest pain or severe breathlessness",
        "Progressive neurological weakness or difficulty walking",
        "Severe persistent headache, significant swelling, unusual bleeding or bruising",
    ]:
        st.markdown(f"• {item}")

    st.markdown("### Model-specific guidance")
    st.info("For a predicted reaction, use the guidance shown in the Predict tab. Guidance is intentionally conservative and should not replace professional medical assessment.")

# ============================================================
# PROJECT
# ============================================================
with tab_project:
    st.markdown("## ℹ️ Project")
    st.markdown("### Vaccine Reaction Prediction Framework")
    st.write("This application uses the trained model and the compiled AEFI dataset to estimate the three highest-probability reaction categories for the entered features.")

    st.markdown("### Current model configuration")
    model_items = {
        "Model": metadata.get("model_type", "Not specified"),
        "Prediction classes": metadata.get("n_classes", metadata.get("num_classes", len(getattr(model, "classes_", [])))),
        "Threshold": metadata.get("threshold", "Not specified"),
        "Top-N predictions": TOP_N_REACTIONS,
        "Date input/features": "Removed",
        "Reason/Outcome feature": "Removed",
        "AEFI grade source": GRADE_COLUMN,
    }
    perf_keys = [
        ("Accuracy", "accuracy"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
    ]
    for label, key in perf_keys:
        if key in metadata:
            model_items[label] = metadata[key]

    st.dataframe(pd.DataFrame([model_items]), use_container_width=True, hide_index=True)

    st.markdown("### Input features")
    st.write("Vaccine, age, sex, age group, vaccine-age interaction, and vaccine-sex interaction are used by the current model input pipeline. Reason for Reporting / Outcome is not used as an input.")

    st.markdown("### AEFI Grade / Classification")
    st.write("For each of the top three predicted reactions, the app displays the most common historical value of 'CLASSIFICATION* BY NATIONAL AEFI COMMITTEE' for that reaction in the supplied Excel dataset.")

    st.markdown("### Important limitations")
    st.write("Model probabilities reflect patterns learned from the dataset. They should not be interpreted as individual patient risk, diagnosis, causality, or a substitute for clinical judgment.")
