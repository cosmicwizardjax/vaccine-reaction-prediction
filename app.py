import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# FILES
# ============================================================

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "reaction_model.pkl"
METADATA_FILE = BASE / "model_metadata.pkl"
STATS_FILE = BASE / "dashboard_stats.json"

OTHER_LABEL = "OTHER REPORTED REACTION (PRESENT IN DATASET)"

# Historical ranges represented in the project dataset.
DATA_MIN_AGE, DATA_MAX_AGE = 12, 97
DATA_MIN_DATE, DATA_MAX_DATE = date(2021, 1, 3), date(2022, 12, 2)

st.set_page_config(
    page_title="Vaccine Reaction Prediction",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
.block-container {max-width: 1280px; padding-top: 1.15rem;}
.hero {
  padding: 1.8rem 1.9rem;
  border-radius: 24px;
  background: linear-gradient(135deg,#08111f 0%,#12395b 52%,#17768d 100%);
  color:#fff; margin-bottom:1rem;
  box-shadow:0 14px 38px rgba(15,23,42,.16);
}
.hero h1 {margin:0;font-size:2.55rem;letter-spacing:-.035em;}
.hero p {margin:.5rem 0 0;opacity:.9;font-size:1.04rem;}
.pred-main {
  padding: 1.55rem 1.5rem;
  border-radius: 22px;
  background: linear-gradient(135deg,#f7fbff,#eaf7f8);
  border:1px solid #c9e4ea;
  margin:.75rem 0 1rem;
  box-shadow:0 8px 24px rgba(15,23,42,.06);
  text-align:center;
}
.pred-main .label {font-size:.82rem;text-transform:uppercase;letter-spacing:.12em;color:#64748b;}
.pred-main .reaction {font-size:2rem;font-weight:800;line-height:1.2;margin-top:.4rem;}
.caution {
  padding:1rem 1.15rem;border-radius:16px;
  background:#fffbeb;border:1px solid #fde68a;
  color:#713f12;margin:.65rem 0;
}
.alert-red {padding:1rem 1.15rem;border-radius:15px;background:#fff1f2;border:1px solid #fecdd3;}
.alert-amber {padding:1rem 1.15rem;border-radius:15px;background:#fffbeb;border:1px solid #fde68a;}
.alert-green {padding:1rem 1.15rem;border-radius:15px;background:#f0fdf4;border:1px solid #bbf7d0;}
.small {font-size:.88rem;color:#64748b;}
.metric-card {
  padding:1rem;border-radius:16px;border:1px solid #dbe4ee;
  background:#fff;text-align:center;box-shadow:0 4px 15px rgba(15,23,42,.04);
}
.metric-card .num {font-size:1.55rem;font-weight:800;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL / DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_metadata():
    return joblib.load(METADATA_FILE)

@st.cache_data
def load_stats():
    if STATS_FILE.exists():
        return json.loads(STATS_FILE.read_text(encoding="utf-8"))
    return {}

model = load_model()
metadata = load_metadata()
stats = load_stats()

# ============================================================
# FEATURES
# ============================================================

def age_group(age):
    if age < 18:
        return "UNDER 18"
    if age <= 30:
        return "18-30"
    if age <= 44:
        return "31-44"
    if age <= 59:
        return "45-59"
    return "60+"

def make_input(age, sex, vaccine, vaccination_date):
    d = pd.Timestamp(vaccination_date)
    ag = age_group(age)

    return pd.DataFrame([{
        "AGE (IN YEARS)": float(age),
        "SEX": sex,
        "VACCINE": vaccine,
        "AGE_GROUP": ag,
        "VACCINATION_YEAR": str(d.year),
        "VACCINATION_MONTH": str(d.month),
        "VACCINATION_QUARTER": str(d.quarter),
        "VACCINE_AGE_GROUP": f"{vaccine}_{ag}",
        "VACCINE_SEX": f"{vaccine}_{sex}",
    }])

# ============================================================
# PREDICTION
# ============================================================

def get_specific_prediction(X):
    """
    The training dataset contains a very large OTHER class.
    For the patient-facing result we deliberately exclude that
    catch-all class and select the highest-scoring specific
    reaction category.

    This is NOT a probability of the patient developing the event.
    It is simply the model's closest specific category among the
    categories represented separately in the training data.
    """
    probs = model.predict_proba(X)[0]
    classes = list(model.classes_)

    specific = [
        (str(label), float(prob))
        for label, prob in zip(classes, probs)
        if str(label).strip() != OTHER_LABEL
    ]

    specific.sort(key=lambda x: x[1], reverse=True)

    if not specific:
        return "REACTION CATEGORY NOT AVAILABLE", 0.0, []

    top_label, top_score = specific[0]
    return top_label.strip(), top_score, specific

# ============================================================
# SAFETY / GUIDANCE
# ============================================================

def safety_alerts(age, vaccine, vaccination_date, reaction):
    alerts = []

    if age < DATA_MIN_AGE or age > DATA_MAX_AGE:
        alerts.append((
            "amber",
            f"Age {age} is outside the dataset's observed range "
            f"({DATA_MIN_AGE}-{DATA_MAX_AGE} years). The model is extrapolating."
        ))

    if vaccination_date < DATA_MIN_DATE or vaccination_date > DATA_MAX_DATE:
        alerts.append((
            "amber",
            "The vaccination date is outside the historical date range "
            "represented in this project dataset. Interpret the result cautiously."
        ))

    r = reaction.upper()
    severe_terms = [
        "ANAPHYLAXIS", "SUDDEN CARDIAC", "MYOCARDIAL", "CORONARY",
        "THROMBOSIS", "THROMBOCYTOPENIA", "GUILLAIN",
        "CEREBROVASCULAR", "SEIZURE", "UNEXPLAINED DEATH"
    ]

    if any(x in r for x in severe_terms):
        alerts.append((
            "red",
            "This category can represent a potentially serious event. "
            "If the person currently has severe or rapidly worsening symptoms, "
            "seek urgent medical assessment rather than relying on this tool."
        ))

    # Always provide one simple, non-alarming caution.
    alerts.append((
        "amber",
        "This result is based only on patterns in the project's historical dataset. "
        "The model has done its best to identify the closest specific category, "
        "but this is not a diagnosis and should not replace advice from a healthcare professional."
    ))

    return alerts

def guidance(reaction):
    r = reaction.upper()

    if "ANAPHYLAXIS" in r or "ALLERGIC" in r:
        return (
            "Urgent medical assessment is appropriate for breathing difficulty, "
            "swelling of the face/throat, widespread hives, collapse, or rapidly worsening symptoms."
        )

    if "CARDIAC" in r or "MYOCARDIAL" in r or "CORONARY" in r:
        return (
            "Chest pain, severe breathlessness, fainting, or sudden deterioration "
            "warrants urgent medical evaluation."
        )

    if "THROMBOSIS" in r or "THROMBOCYTOPENIA" in r:
        return (
            "Seek prompt medical assessment for severe headache, new neurological symptoms, "
            "chest pain, breathlessness, leg swelling/pain, or unusual bleeding."
        )

    if "GUILLAIN" in r or "PALSY" in r or "CEREBROVASCULAR" in r or "SEIZURE" in r:
        return (
            "New weakness, facial drooping, difficulty speaking/walking, seizure, "
            "or rapidly progressing neurological symptoms require urgent assessment."
        )

    if "FEVER" in r or "FEBRILE" in r:
        return (
            "Rest, maintain fluids, monitor symptoms, and contact a healthcare professional "
            "if symptoms are severe, persistent, or worsening."
        )

    if "ANXIETY" in r or "VASOVAGAL" in r or "CONVERSION" in r:
        return (
            "Sit or lie down safely, avoid driving while symptomatic, and seek assessment "
            "if fainting, persistent symptoms, or severe symptoms occur."
        )

    return (
        "The correct treatment depends on the actual clinical diagnosis. "
        "A healthcare professional should evaluate the symptoms before treatment is selected."
    )

def resource_links(age, vaccine, reaction):
    links = [
        (
            "WHO — Vaccine safety",
            "https://www.who.int/news-room/questions-and-answers/item/vaccines-and-immunization-vaccine-safety"
        ),
        (
            "WHO — Vaccines and immunization",
            "https://www.who.int/health-topics/vaccines-and-immunization/"
        ),
    ]

    if "ANAPHYLAXIS" in reaction.upper() or "ALLERGIC" in reaction.upper():
        links.insert(
            0,
            (
                "WHO — Vaccine safety information",
                "https://www.who.int/news-room/questions-and-answers/item/vaccines-and-immunization-vaccine-safety"
            )
        )

    if age >= 60:
        links.append(
            (
                "WHO — Vaccines and immunization",
                "https://www.who.int/health-topics/vaccines-and-immunization/"
            )
        )

    return links

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
<h1>💉 Vaccine Reaction Prediction</h1>
<p>Dataset-based prediction • interactive analytics • personalized safety guidance • offline-ready</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "This application provides a dataset-based prediction category. "
    "It is not a diagnosis and does not prove that a vaccine caused an event. "
    "Do not use it as a substitute for professional medical care."
)

tabs = st.tabs([
    "🔎 Predict",
    "📊 Interactive Analytics",
    "🛡️ Safety & Resources",
    "ℹ️ Project"
])

# ============================================================
# PREDICTION TAB
# ============================================================

with tabs[0]:
    st.subheader("Enter patient & vaccination details")

    c1, c2, c3 = st.columns(3)

    with c1:
        vaccine = st.selectbox(
            "Vaccine",
            ["COVISHIELD", "COVAXIN", "CORBEVAX", "SPUTNIK V"]
        )

    with c2:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=45,
            step=1
        )

    with c3:
        sex = st.selectbox("Sex", ["MALE", "FEMALE"])

    vaccination_date = st.date_input(
        "Vaccination date",
        value=date(2022, 1, 1),
        min_value=date(2020, 1, 1),
        max_value=date.today()
    )

    if st.button(
        "✨ Predict Possible Reaction",
        type="primary",
        use_container_width=True
    ):
        X = make_input(age, sex, vaccine, vaccination_date)
        reaction, internal_score, all_specific = get_specific_prediction(X)

        st.session_state.reaction = reaction
        st.session_state.internal_score = internal_score
        st.session_state.input = {
            "age": age,
            "sex": sex,
            "vaccine": vaccine,
            "date": vaccination_date,
        }

    if "reaction" in st.session_state:
        reaction = st.session_state.reaction
        inp = st.session_state.input

        st.subheader("Prediction")

        st.markdown(
            f"""
            <div class="pred-main">
                <div class="label">Dataset-based model result</div>
                <div class="reaction">{reaction} MAY OCCUR</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Deliberately do not display model accuracy or probability scores.
        for level, msg in safety_alerts(
            inp["age"], inp["vaccine"], inp["date"], reaction
        ):
            css = {
                "red": "alert-red",
                "amber": "alert-amber",
                "green": "alert-green"
            }[level]

            icon = "🚨" if level == "red" else "⚠️" if level == "amber" else "✓"

            st.markdown(
                f'<div class="{css}">{icon} {msg}</div>',
                unsafe_allow_html=True
            )

        st.subheader("General precautions")
        st.write(guidance(reaction))

        st.caption(
            "The application does not prescribe medicines or doses. "
            "Treatment should be based on a clinician's assessment."
        )

        st.subheader("Relevant official resources")
        for label, url in resource_links(
            inp["age"], inp["vaccine"], reaction
        ):
            st.markdown(f"🔗 [{label}]({url})")

        report = f"""Vaccine Reaction Prediction Report

Inputs:
Age: {inp["age"]}
Sex: {inp["sex"]}
Vaccine: {inp["vaccine"]}
Vaccination date: {inp["date"]}

Model result:
{reaction} MAY OCCUR

Caution:
This is a dataset-based model result, not a medical diagnosis.
Please consult a healthcare professional and do not rely on this result alone.
"""

        st.download_button(
            "⬇️ Save result summary",
            report,
            "vaccine_reaction_result.txt",
            "text/plain"
        )

# ============================================================
# ANALYTICS TAB
# ============================================================

with tabs[1]:
    st.subheader("Interactive dataset analytics")
    st.caption(
        "These charts use aggregate project statistics. "
        "They do not display individual patient records."
    )

    vaccine_counts = stats.get("vaccine_counts", {})
    v = pd.DataFrame({
        "Vaccine": list(vaccine_counts.keys()),
        "Records": list(vaccine_counts.values())
    })

    if not v.empty:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.pie(
                v,
                names="Vaccine",
                values="Records",
                hole=.55,
                title="Vaccine distribution"
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.treemap(
                v,
                path=["Vaccine"],
                values="Records",
                title="Vaccine volume"
            )
            st.plotly_chart(fig, use_container_width=True)

    diagnosis_counts = stats.get("diagnosis_counts", {})
    specific_diagnoses = {
        k: v for k, v in diagnosis_counts.items()
        if str(k).strip() != OTHER_LABEL
    }

    d = pd.DataFrame({
        "Reaction": list(specific_diagnoses.keys()),
        "Records": list(specific_diagnoses.values())
    })

    if not d.empty:
        st.markdown("### Specific reaction categories in the dataset")

        fig = px.bar(
            d.sort_values("Records"),
            x="Records",
            y="Reaction",
            orientation="h",
            title="Specific reaction categories",
            text="Records"
        )
        fig.update_layout(yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            fig = px.pie(
                d.sort_values("Records", ascending=False).head(8),
                names="Reaction",
                values="Records",
                hole=.45,
                title="Top specific categories"
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.treemap(
                d.sort_values("Records", ascending=False),
                path=["Reaction"],
                values="Records",
                title="Reaction-category treemap"
            )
            st.plotly_chart(fig, use_container_width=True)

    age_counts = stats.get("age_group_counts", {})
    a = pd.DataFrame({
        "Age group": list(age_counts.keys()),
        "Records": list(age_counts.values())
    })

    if not a.empty:
        fig = px.pie(
            a,
            names="Age group",
            values="Records",
            hole=.45,
            title="Age-group distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    heat = stats.get("vaccine_sex", {})
    if heat:
        h = pd.DataFrame(heat).T.fillna(0)

        fig = px.imshow(
            h,
            text_auto=True,
            aspect="auto",
            title="Vaccine × sex distribution",
            labels={"x": "Sex", "y": "Vaccine", "color": "Records"}
        )
        st.plotly_chart(fig, use_container_width=True)

    years = stats.get("vaccination_year_counts", {})
    if years:
        ydf = pd.DataFrame({
            "Year": [str(k) for k in years],
            "Records": [years[k] for k in years]
        })

        fig = go.Figure(
            go.Scatter(
                x=ydf["Year"],
                y=ydf["Records"],
                mode="lines+markers",
                hovertemplate="Year: %{x}<br>Records: %{y}<extra></extra>"
            )
        )

        fig.update_layout(
            title="Vaccination records by year",
            xaxis_title="Year",
            yaxis_title="Records"
        )

        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SAFETY TAB
# ============================================================

with tabs[2]:
    st.subheader("Safety alerts & trusted resources")

    if "reaction" in st.session_state:
        inp = st.session_state.input
        reaction = st.session_state.reaction

        for level, msg in safety_alerts(
            inp["age"], inp["vaccine"], inp["date"], reaction
        ):
            css = {
                "red": "alert-red",
                "amber": "alert-amber",
                "green": "alert-green"
            }[level]

            st.markdown(
                f'<div class="{css}">{msg}</div>',
                unsafe_allow_html=True
            )

        st.markdown("### Resources relevant to this result")

        for label, url in resource_links(
            inp["age"], inp["vaccine"], reaction
        ):
            st.markdown(f"🔗 [{label}]({url})")

    else:
        st.write(
            "Run a prediction first to generate input-specific safety alerts "
            "and resource suggestions."
        )

    st.markdown("### Emergency warning signs")
    st.write(
        "Seek urgent care for severe breathing difficulty, swelling of the "
        "face/throat, collapse, severe chest pain, seizure, new major neurological "
        "symptoms, or rapidly worsening symptoms."
    )

# ============================================================
# ABOUT TAB
# ============================================================

with tabs[3]:
    st.subheader("About this application")

    st.write("Model:", metadata.get("model_type", "—"))
    st.write("Prediction classes:", metadata.get("number_of_classes", "—"))
    st.write("Training threshold:", metadata.get("threshold", "—"))
    st.write(
        "Training data age range:",
        f"{DATA_MIN_AGE}–{DATA_MAX_AGE} years"
    )
    st.write(
        "Training vaccination-date range:",
        f"{DATA_MIN_DATE} to {DATA_MAX_DATE}"
    )

    st.info(
        "The patient-facing interface intentionally does not display "
        "model accuracy or probability scores."
    )

    st.markdown("### Important limitation")
    st.write(
        "The application identifies the closest specific reaction category "
        "represented in the project's historical data. It cannot establish "
        "whether vaccination caused an event and it cannot replace clinical "
        "evaluation."
    )

    st.markdown("### Why an AEFI is not automatically caused by a vaccine")
    st.write(
        "An adverse event following immunization is an event occurring after "
        "vaccination; it does not by itself establish causation. Causality "
        "requires clinical and surveillance assessment."
    )

    st.markdown(
        "[WHO vaccine safety information]"
        "(https://www.who.int/news-room/questions-and-answers/item/"
        "vaccines-and-immunization-vaccine-safety)"
    )

    st.markdown(
        "[India Ministry of Health & Family Welfare — AEFI surveillance guidelines]"
        "(https://www.mohfw.gov.in/sites/default/files/"
        "National%20AEFI%20Surveillance%20and%20Response%20Operational%20Guidelines%202024.pdf)"
    )
