from pathlib import Path
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "titanic.csv"
MODEL_PATHS = {
    "Logistic Regression": PROJECT_DIR / "titanic_logisticregression.pkl",
    "SVM": PROJECT_DIR / "titanic_svm_model.pkl",
    "KNN": PROJECT_DIR / "titani_knn_model.pkl",
}
FEATURES = [
    "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone",
    "Sex_male", "Embarked_Q", "Embarked_S",
]


def prepare_dataset(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned["Age"] = cleaned["Age"].fillna(cleaned["Age"].median())
    cleaned = cleaned.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    cleaned["Embarked"] = cleaned["Embarked"].fillna(cleaned["Embarked"].mode()[0])
    cleaned["FamilySize"] = cleaned["SibSp"] + cleaned["Parch"] + 1
    cleaned["IsAlone"] = (cleaned["FamilySize"] == 1).astype(int)
    return pd.get_dummies(cleaned, columns=["Sex", "Embarked"], drop_first=True)


@st.cache_resource
def load_dashboard_data():
    raw_data = pd.read_csv(DATA_PATH)
    prepared_data = prepare_dataset(raw_data)
    features = prepared_data.drop(columns="Survived").reindex(columns=FEATURES, fill_value=0)
    labels = prepared_data["Survived"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.20, random_state=42
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
    for name, model in models.items():
        if getattr(model, "n_features_in_", len(FEATURES)) != len(FEATURES):
            raise ValueError(f"{name} expects a different number of input features.")
    return raw_data, x_test_scaled, y_test, scaler, models


def make_passenger(pclass, sex, age, siblings, parents, fare, embarked):
    passenger = pd.DataFrame([{ 
        "Pclass": pclass,
        "Sex_male": int(sex == "Male"),
        "Age": age,
        "SibSp": siblings,
        "Parch": parents,
        "Fare": fare,
        "Embarked_Q": int(embarked == "Queenstown (Q)"),
        "Embarked_S": int(embarked == "Southampton (S)"),
        "FamilySize": siblings + parents + 1,
        "IsAlone": int(siblings + parents == 0),
    }], columns=FEATURES)
    return passenger


st.set_page_config(page_title="Titanic Model Explorer", page_icon="🚢", layout="wide")
st.title("Titanic Model Explorer")
st.caption("Explore passenger outcomes, compare saved classifiers, and test a passenger profile.")

try:
    raw_data, x_test_scaled, y_test, scaler, models = load_dashboard_data()
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(f"Could not load the Titanic data or model files: {error}")
    st.stop()

survival_rate = raw_data["Survived"].mean() * 100
survivors = int(raw_data["Survived"].sum())
passenger_count = len(raw_data)

metric_columns = st.columns(3)
metric_columns[0].metric("Passengers", f"{passenger_count:,}")
metric_columns[1].metric("Survivors", f"{survivors:,}")
metric_columns[2].metric("Overall survival", f"{survival_rate:.1f}%")

overview_tab, performance_tab, prediction_tab = st.tabs(
    ["Passenger overview", "Model performance", "Try a passenger"]
)

with overview_tab:
    st.subheader("Survival patterns")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        survival_by_sex = raw_data.groupby("Sex")["Survived"].mean().mul(100)
        st.markdown("**Survival rate by sex**")
        st.bar_chart(survival_by_sex, y_label="Survival rate (%)")
    with chart_right:
        survival_by_class = raw_data.groupby("Pclass")["Survived"].mean().mul(100)
        st.markdown("**Survival rate by passenger class**")
        st.bar_chart(survival_by_class, y_label="Survival rate (%)")

    age_data = raw_data.assign(AgeBand=pd.cut(raw_data["Age"], bins=[0, 12, 18, 30, 50, 80],
                                               labels=["0-12", "13-18", "19-30", "31-50", "51+"]))
    age_survival = age_data.groupby(["AgeBand", "Survived"], observed=False).size().unstack(fill_value=0)
    age_survival.columns = ["Did not survive", "Survived"]
    st.markdown("**Passenger count by age band**")
    st.bar_chart(age_survival)

with performance_tab:
    st.subheader("Held-out test performance")
    st.caption("Scores use the original 80/20 split (random_state=42) and preprocessing from the training script.")
    score_rows = []
    for name, model in models.items():
        predictions = model.predict(x_test_scaled)
        score_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 score": f1_score(y_test, predictions, zero_division=0),
        })
    scores = pd.DataFrame(score_rows).set_index("Model")
    st.markdown("**Accuracy by model**")
    st.bar_chart(scores["Accuracy"], y_label="Accuracy")
    st.markdown("**Detailed scores**")
    st.dataframe(scores.style.format("{:.3f}"), use_container_width=True)

    selected_model = st.selectbox("Confusion matrix", list(models))
    selected_predictions = models[selected_model].predict(x_test_scaled)
    matrix = pd.DataFrame(
        confusion_matrix(y_test, selected_predictions, labels=[0, 1]),
        index=["Actual: did not survive", "Actual: survived"],
        columns=["Predicted: did not survive", "Predicted: survived"],
    )
    st.dataframe(matrix, use_container_width=True)

with prediction_tab:
    st.subheader("Passenger profile")
    with st.form("passenger_form"):
        input_left, input_right = st.columns(2)
        with input_left:
            pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
            sex = st.selectbox("Sex", ["Female", "Male"])
            age = st.slider("Age", min_value=0, max_value=80, value=25)
            fare = st.number_input("Fare", min_value=0.0, value=7.25, step=1.0)
        with input_right:
            siblings = st.number_input("Siblings / spouses aboard", min_value=0, max_value=8, value=0)
            parents = st.number_input("Parents / children aboard", min_value=0, max_value=6, value=0)
            embarked = st.selectbox("Port of embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
            chosen_model = st.selectbox("Classifier", list(models))
        submitted = st.form_submit_button("Predict survival")

    if submitted:
        passenger = make_passenger(pclass, sex, age, siblings, parents, fare, embarked)
        passenger_scaled = scaler.transform(passenger)
        model = models[chosen_model]
        prediction = int(model.predict(passenger_scaled)[0])
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(passenger_scaled)[0][1])
            outcome = "survive" if prediction == 1 else "not survive"
            message = f"Predicted to {outcome} · {probability:.1%} survival probability"
        else:
            outcome = "survive" if prediction == 1 else "not survive"
            message = f"Predicted to {outcome} · this model does not provide probability estimates"
        if prediction == 1:
            st.success(message)
        else:
            st.warning(message)

st.caption("Model predictions are estimates, not historical certainty.")
