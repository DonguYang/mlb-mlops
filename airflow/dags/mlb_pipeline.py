"""Airflow DAG: 매일 MLB 데이터 수집 → 피처 생성 → 모델 재학습"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
sys.path.insert(0, "/opt/airflow")

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="mlb_daily_pipeline",
    default_args=default_args,
    schedule="0 9 * * *",  # 매일 오전 9시 (KST 기준 설정 필요)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlb", "mlops"],
) as dag:

    def collect_task():
        from src.data.collect import collect_seasons
        from datetime import date
        current_year = date.today().year
        collect_seasons(start=current_year, end=current_year)

    def features_task():
        from src.data.features import build_features
        build_features()

    def train_task():
        import subprocess
        subprocess.run(
            ["python", "-m", "src.models.train", "--model", "all"],
            cwd="/opt/airflow",
            check=True,
        )

    collect = PythonOperator(task_id="collect_data", python_callable=collect_task)
    features = PythonOperator(task_id="build_features", python_callable=features_task)
    train = PythonOperator(task_id="train_model", python_callable=train_task)

    collect >> features >> train
