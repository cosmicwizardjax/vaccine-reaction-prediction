import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from catboost import CatBoostClassifier


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/final ugrc(1).xlsx"

SHEET_NAME = "compiled data all"

RANDOM_STATE = 42


# Minimum number of records required to keep a diagnosis
# as an individual prediction class.
THRESHOLDS = [15, 20, 25]


OTHER_LABEL = (
    "OTHER REPORTED REACTION "
    "(PRESENT IN DATASET)"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("VACCINE REACTION MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET_NAME
)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

print(
    "Original records:",
    len(df)
)


# ============================================================
# CLEAN SEX
# ============================================================

print("\nCleaning SEX column...")

df["SEX"] = (
    df["SEX"]
    .astype("string")
    .str.replace(
        "\n",
        "",
        regex=False
    )
    .str.strip()
    .str.upper()
)

df["SEX"] = df["SEX"].replace({

    "M": "MALE",

    "F": "FEMALE",

    "FEMAL E": "FEMALE"
})

df["SEX"] = (
    df["SEX"]
    .fillna("UNKNOWN")
)


# ============================================================
# CLEAN VACCINE
# ============================================================

print("Cleaning VACCINE column...")

df["VACCINE"] = (
    df["VACCINE"]
    .astype("string")
    .str.replace(
        "\n",
        "",
        regex=False
    )
    .str.strip()
    .str.upper()
)

df["VACCINE"] = df["VACCINE"].replace({

    "COVISHIELDPAGE 1 OF 8":
        "COVISHIELD",

    "CORBEVAXAGE 8 OF 12":
        "CORBEVAX"
})

df["VACCINE"] = (
    df["VACCINE"]
    .fillna("UNKNOWN")
)


# ============================================================
# CLEAN AGE
# ============================================================

print("Cleaning AGE column...")

df["AGE (IN YEARS)"] = pd.to_numeric(
    df["AGE (IN YEARS)"],
    errors="coerce"
)


# ============================================================
# AGE GROUP
# ============================================================

def make_age_group(age):

    if pd.isna(age):

        return "UNKNOWN"

    if age < 18:

        return "UNDER 18"

    elif age <= 30:

        return "18-30"

    elif age <= 44:

        return "31-44"

    elif age <= 59:

        return "45-59"

    else:

        return "60+"


df["AGE_GROUP"] = (
    df["AGE (IN YEARS)"]
    .apply(make_age_group)
)


# ============================================================
# CLEAN VACCINATION DATE
# ============================================================

print("Processing vaccination dates...")

dates = pd.to_datetime(
    df["DATE OF VACCINATION (DD/MM/YYYY)"],
    errors="coerce",
    dayfirst=True
)


# Remove impossible old dates
dates.loc[
    dates.dt.year < 2020
] = pd.NaT


df["VACCINATION_YEAR"] = (
    dates.dt.year
    .fillna(0)
    .astype(int)
    .astype(str)
)


df["VACCINATION_MONTH"] = (
    dates.dt.month
    .fillna(0)
    .astype(int)
    .astype(str)
)


df["VACCINATION_QUARTER"] = (
    dates.dt.quarter
    .fillna(0)
    .astype(int)
    .astype(str)
)


print("\nVaccination date information:")

print(
    "Valid dates:",
    dates.notna().sum()
)

print(
    "Invalid/missing dates:",
    dates.isna().sum()
)

if dates.notna().any():

    print(
        "Earliest date:",
        dates.min()
    )

    print(
        "Latest date:",
        dates.max()
    )


# ============================================================
# CREATE INTERACTION FEATURES
# ============================================================

print("\nCreating interaction features...")


df["VACCINE_AGE_GROUP"] = (

    df["VACCINE"].astype(str)

    + "_"

    + df["AGE_GROUP"].astype(str)
)


df["VACCINE_SEX"] = (

    df["VACCINE"].astype(str)

    + "_"

    + df["SEX"].astype(str)
)


# ============================================================
# CLEAN DIAGNOSIS
# ============================================================

print("Cleaning DIAGNOSIS column...")


df["DIAGNOSIS"] = (

    df["DIAGNOSIS"]

    .astype("string")

    .str.replace(
        "\n",
        " ",
        regex=False
    )

    .str.strip()

    .str.upper()
)


df["DIAGNOSIS"] = df["DIAGNOSIS"].replace(
    "",
    pd.NA
)


# ============================================================
# MERGE DUPLICATE DIAGNOSIS SPELLINGS
# ============================================================

df["DIAGNOSIS"] = df["DIAGNOSIS"].replace({

    "GUILLIAN BARRE SYNDROME":
        "GUILLAIN BARRE SYNDROME",

    "GUILLAIN-BARRE SYNDROME":
        "GUILLAIN BARRE SYNDROME",

    "COVID-19 DISEASE":
        "COVID 19 DISEASE",

    "COVID-19 PNEUMONIA":
        "COVID 19 PNEUMONIA",

    "BELL’S PALSY":
        "BELL'S PALSY",

    "SUDDEN CARDIAC DEATH IN KNOWN CASE OF HYPERTENSION":
        "SUDDEN CARDIAC DEATH IN A KNOWN CASE OF HYPERTENSION",

    "FEVER WITH VOMITING":
        "FEVER AND VOMITING"
})


# ============================================================
# REMOVE MISSING TARGETS
# ============================================================

df_ml = df[
    df["DIAGNOSIS"].notna()
].copy()


print(
    "\nRecords available for ML:",
    len(df_ml)
)


print(
    "Unique original diagnoses:",
    df_ml["DIAGNOSIS"].nunique()
)


# ============================================================
# CREATE TARGET
# ============================================================

def make_target(
    data,
    threshold
):

    counts = (
        data["DIAGNOSIS"]
        .value_counts()
    )


    common_classes = counts[
        counts >= threshold
    ].index


    target = data[
        "DIAGNOSIS"
    ].apply(

        lambda diagnosis:

        diagnosis

        if diagnosis in common_classes

        else OTHER_LABEL
    )


    return target


# ============================================================
# FEATURES
# ============================================================

features = [

    "AGE (IN YEARS)",

    "SEX",

    "VACCINE",

    "AGE_GROUP",

    "VACCINATION_YEAR",

    "VACCINATION_MONTH",

    "VACCINATION_QUARTER",

    "VACCINE_AGE_GROUP",

    "VACCINE_SEX"
]


categorical_features = [

    "SEX",

    "VACCINE",

    "AGE_GROUP",

    "VACCINATION_YEAR",

    "VACCINATION_MONTH",

    "VACCINATION_QUARTER",

    "VACCINE_AGE_GROUP",

    "VACCINE_SEX"
]


# ============================================================
# PREPARE FEATURES
# ============================================================

print("\nPreparing ML features...")


X = df_ml[
    features
].copy()


# Clean categorical values
for column in categorical_features:

    X[column] = (

        X[column]

        .fillna("UNKNOWN")

        .astype(str)
    )


# Clean numerical age
X["AGE (IN YEARS)"] = pd.to_numeric(

    X["AGE (IN YEARS)"],

    errors="coerce"
)


age_median = X[
    "AGE (IN YEARS)"
].median()


X["AGE (IN YEARS)"] = (

    X["AGE (IN YEARS)"]

    .fillna(age_median)
)


# ============================================================
# EXTRA TREES PREPROCESSOR
# ============================================================

extra_trees_preprocessor = ColumnTransformer(

    transformers=[

        (

            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        )
    ],

    remainder="passthrough"
)


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(

    n_splits=3,

    shuffle=True,

    random_state=RANDOM_STATE
)


# ============================================================
# EVALUATE MODEL FUNCTION
# ============================================================

def evaluate_model(

    model,

    X,

    y,

    model_name,

    categorical_indices=None
):

    accuracies = []

    macro_f1_scores = []

    weighted_f1_scores = []


    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):


        print(
            f"    Fold {fold}/3...",
            end=" ",
            flush=True
        )


        X_train = X.iloc[
            train_idx
        ].copy()


        X_test = X.iloc[
            test_idx
        ].copy()


        y_train = y.iloc[
            train_idx
        ].copy()


        y_test = y.iloc[
            test_idx
        ].copy()


        # ----------------------------------------------------
        # FIT MODEL
        # ----------------------------------------------------

        if model_name == "CatBoost":

            model.fit(

                X_train,

                y_train,

                cat_features=categorical_indices
            )

        else:

            model.fit(

                X_train,

                y_train
            )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )


        predictions = np.asarray(
            predictions
        ).ravel()


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(

            y_test,

            predictions
        )


        macro_f1 = f1_score(

            y_test,

            predictions,

            average="macro",

            zero_division=0
        )


        weighted_f1 = f1_score(

            y_test,

            predictions,

            average="weighted",

            zero_division=0
        )


        accuracies.append(
            accuracy
        )


        macro_f1_scores.append(
            macro_f1
        )


        weighted_f1_scores.append(
            weighted_f1
        )


        print(
            f"Accuracy={accuracy:.4f} "
            f"MacroF1={macro_f1:.4f}"
        )


    return {

        "accuracy":
            float(np.mean(accuracies)),

        "macro_f1":
            float(np.mean(macro_f1_scores)),

        "weighted_f1":
            float(np.mean(weighted_f1_scores))
    }


# ============================================================
# RUN MODEL EXPERIMENTS
# ============================================================

results = []


categorical_indices = [

    features.index(column)

    for column in categorical_features
]


for threshold in THRESHOLDS:


    print("\n" + "=" * 70)

    print(
        f"TESTING THRESHOLD: {threshold}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # CREATE TARGET
    # --------------------------------------------------------

    y = make_target(

        df_ml,

        threshold
    )


    class_count = y.nunique()


    other_count = (

        y == OTHER_LABEL

    ).sum()


    print(
        "Prediction classes:",
        class_count
    )


    print(
        "Records grouped as OTHER:",
        other_count
    )


    # ========================================================
    # EXTRA TREES
    # ========================================================

    print(
        "\n  Testing Extra Trees..."
    )


    extra_trees_classifier = ExtraTreesClassifier(

        n_estimators=300,

        max_features="sqrt",

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


    extra_trees_model = Pipeline(

        steps=[

            (

                "preprocessor",

                extra_trees_preprocessor
            ),

            (

                "classifier",

                extra_trees_classifier
            )
        ]
    )


    scores = evaluate_model(

        model=extra_trees_model,

        X=X,

        y=y,

        model_name="Extra Trees"
    )


    print(
        f"  Accuracy: {scores['accuracy']:.4f}"
    )


    print(
        f"  Macro F1: {scores['macro_f1']:.4f}"
    )


    print(
        f"  Weighted F1: {scores['weighted_f1']:.4f}"
    )


    results.append({

        "threshold": threshold,

        "model": "Extra Trees",

        "classes": class_count,

        **scores
    })


    # ========================================================
    # CATBOOST
    # ========================================================

    print(
        "\n  Testing CatBoost..."
    )


    catboost_model = CatBoostClassifier(

        iterations=300,

        depth=6,

        learning_rate=0.08,

        loss_function="MultiClass",

        random_seed=RANDOM_STATE,

        verbose=False,

        allow_writing_files=False,

        thread_count=-1
    )


    scores = evaluate_model(

        model=catboost_model,

        X=X,

        y=y,

        model_name="CatBoost",

        categorical_indices=categorical_indices
    )


    print(
        f"  Accuracy: {scores['accuracy']:.4f}"
    )


    print(
        f"  Macro F1: {scores['macro_f1']:.4f}"
    )


    print(
        f"  Weighted F1: {scores['weighted_f1']:.4f}"
    )


    results.append({

        "threshold": threshold,

        "model": "CatBoost",

        "classes": class_count,

        **scores
    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(

    by=[
        "macro_f1",
        "weighted_f1",
        "accuracy"
    ],

    ascending=[
        False,
        False,
        False
    ]
)


print("\n\n" + "=" * 70)

print("MODEL COMPARISON RESULTS")

print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best = results_df.iloc[0]


best_threshold = int(
    best["threshold"]
)


best_model_name = str(
    best["model"]
)


print("\n" + "=" * 70)

print("BEST CONFIGURATION")

print("=" * 70)


print(
    "Model:",
    best_model_name
)


print(
    "Threshold:",
    best_threshold
)


print(
    "Classes:",
    int(best["classes"])
)


print(
    "Accuracy:",
    f"{best['accuracy']:.4f}"
)


print(
    "Macro F1:",
    f"{best['macro_f1']:.4f}"
)


print(
    "Weighted F1:",
    f"{best['weighted_f1']:.4f}"
)


# ============================================================
# CREATE FINAL TARGET
# ============================================================

y_final = make_target(

    df_ml,

    best_threshold
)


print(
    "\nFinal prediction classes:"
)


print(
    y_final.value_counts()
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 70)

print("TRAINING FINAL MODEL")

print("=" * 70)


if best_model_name == "CatBoost":


    # --------------------------------------------------------
    # FINAL CATBOOST MODEL
    # --------------------------------------------------------

    final_model = CatBoostClassifier(

        iterations=500,

        depth=6,

        learning_rate=0.08,

        loss_function="MultiClass",

        random_seed=RANDOM_STATE,

        verbose=False,

        allow_writing_files=False,

        thread_count=-1
    )


    final_model.fit(

        X,

        y_final,

        cat_features=categorical_indices
    )


    final_model.save_model(

        "reaction_model.cbm"
    )


    model_file = (
        "reaction_model.cbm"
    )


    model_classes = list(
        final_model.classes_
    )


else:


    # --------------------------------------------------------
    # FINAL EXTRA TREES MODEL
    # --------------------------------------------------------

    final_classifier = ExtraTreesClassifier(

        n_estimators=500,

        max_features="sqrt",

        min_samples_leaf=2,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


    final_model = Pipeline(

        steps=[

            (

                "preprocessor",

                extra_trees_preprocessor
            ),

            (

                "classifier",

                final_classifier
            )
        ]
    )


    final_model.fit(

        X,

        y_final
    )


    joblib.dump(

        final_model,

        "reaction_model.pkl"
    )


    model_file = (
        "reaction_model.pkl"
    )


    model_classes = list(
        final_model.classes_
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model_type":
        best_model_name,

    "model_file":
        model_file,

    "threshold":
        best_threshold,

    "features":
        features,

    "categorical_features":
        categorical_features,

    "number_of_classes":
        len(model_classes),

    "classes":
        model_classes,

    "accuracy":
        float(best["accuracy"]),

    "macro_f1":
        float(best["macro_f1"]),

    "weighted_f1":
        float(best["weighted_f1"])
}


joblib.dump(

    metadata,

    "model_metadata.pkl"
)


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

results_df.to_csv(

    "model_comparison.csv",

    index=False
)


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 70)

print("TRAINING COMPLETE")

print("=" * 70)


print(
    "Best Model:",
    best_model_name
)


print(
    "Model File:",
    model_file
)


print(
    "Metadata File: model_metadata.pkl"
)


print(
    "Comparison File: model_comparison.csv"
)


print(
    "Number of Prediction Classes:",
    len(model_classes)
)


print(
    "\nTop 3 reaction prediction support: ENABLED"
)


print(
    "\nYou can now run:"
)


print(
    "python -m streamlit run app.py"
)