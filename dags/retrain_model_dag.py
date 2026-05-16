import os
import json
import shutil
import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator


BASE_DIR = os.environ.get("AIRFLOW_HOME", os.getcwd())

DATA_PATH = os.path.join(BASE_DIR, "data", "new_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ARCHIVE_DIR = os.path.join(MODELS_DIR, "archive")
PRODUCTION_DIR = os.path.join(MODELS_DIR, "production")

PRODUCTION_MODEL_PATH = os.path.join(PRODUCTION_DIR, "production_model.pkl")
PRODUCTION_METRICS_PATH = os.path.join(PRODUCTION_DIR, "production_metrics.json")


def prepare_directories():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(PRODUCTION_DIR, exist_ok=True)


def train_new_model(**context):
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    prepare_directories()

    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"rf_model_{timestamp}.pkl"
    metrics_filename = f"rf_model_{timestamp}_metrics.json"

    model_path = os.path.join(ARCHIVE_DIR, model_filename)
    metrics_path = os.path.join(ARCHIVE_DIR, metrics_filename)

    joblib.dump(model, model_path)

    metrics = {
        "model_path": model_path,
        "accuracy": accuracy,
        "created_at": timestamp,
        "algorithm": "RandomForestClassifier",
        "n_estimators": 100,
        "random_state": 42,
        "data_path": DATA_PATH
    }

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    context["ti"].xcom_push(key="new_model_path", value=model_path)
    context["ti"].xcom_push(key="new_metrics_path", value=metrics_path)
    context["ti"].xcom_push(key="new_accuracy", value=accuracy)

    print(f"Nowy model zapisany w: {model_path}")
    print(f"Accuracy nowego modelu: {accuracy}")


def compare_with_production(**context):
    ti = context["ti"]

    new_accuracy = ti.xcom_pull(
        task_ids="train_new_model",
        key="new_accuracy"
    )

    if not os.path.exists(PRODUCTION_METRICS_PATH):
        print("Brak modelu produkcyjnego. Nowy model zostanie wdrożony.")
        return "promote_model_to_production"

    with open(PRODUCTION_METRICS_PATH, "r") as file:
        production_metrics = json.load(file)

    production_accuracy = production_metrics.get("accuracy", 0)

    print(f"Accuracy modelu produkcyjnego: {production_accuracy}")
    print(f"Accuracy nowego modelu: {new_accuracy}")

    if new_accuracy > production_accuracy:
        print("Nowy model jest lepszy. Zostanie wdrożony.")
        return "promote_model_to_production"

    print("Nowy model nie jest lepszy. Pozostaje tylko w archiwum.")
    return "keep_model_archived"


def promote_model_to_production(**context):
    ti = context["ti"]

    new_model_path = ti.xcom_pull(
        task_ids="train_new_model",
        key="new_model_path"
    )

    new_metrics_path = ti.xcom_pull(
        task_ids="train_new_model",
        key="new_metrics_path"
    )

    shutil.copy2(new_model_path, PRODUCTION_MODEL_PATH)
    shutil.copy2(new_metrics_path, PRODUCTION_METRICS_PATH)

    print(f"Model wdrożony do produkcji: {PRODUCTION_MODEL_PATH}")
    print(f"Metryki produkcyjne zapisane w: {PRODUCTION_METRICS_PATH}")


def keep_model_archived():
    print("Model zostaje tylko w archiwum. Produkcja bez zmian.")


default_args = {
    "owner": "student",
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=1),
}


with DAG(
    dag_id="retrain_model_dag",
    description="Automatyczne re-trenowanie, walidacja i wersjonowanie modelu ML",
    default_args=default_args,
    start_date=datetime.datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "retraining", "model-versioning"],
) as dag:

    start = EmptyOperator(task_id="start")

    train_model = PythonOperator(
        task_id="train_new_model",
        python_callable=train_new_model
    )

    compare_models = BranchPythonOperator(
        task_id="compare_with_production",
        python_callable=compare_with_production
    )

    promote_model = PythonOperator(
        task_id="promote_model_to_production",
        python_callable=promote_model_to_production
    )

    archive_model = PythonOperator(
        task_id="keep_model_archived",
        python_callable=keep_model_archived
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success"
    )

    start >> train_model >> compare_models
    compare_models >> promote_model >> end
    compare_models >> archive_model >> end