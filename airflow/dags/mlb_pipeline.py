"""Airflow DAG: 매일 MLB 데이터 수집 → 피처 생성 → 모델 재학습"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="mlb_daily_pipeline",
    default_args=default_args,
    schedule="0 21 * * *",  # 매일 오전 6시 KST (= UTC 21:00)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlb", "mlops"],
) as dag:

    def collect_task():
        """당해 시즌 경기 결과 및 팀 통계(ERA/OPS)를 매일 갱신"""
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.data.collect import update_current_season
        update_current_season()

    def features_task():
        import sys
        sys.path.insert(0, "/opt/airflow")
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
