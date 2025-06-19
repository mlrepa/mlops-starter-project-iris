from datetime import datetime, timedelta

from airflow.models import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

with DAG(
    dag_id="load_and_split_dataset",
    default_args=default_args,
    description="Run load_data.py and split_dataset.py every minute",
    start_date=datetime(2022, 1, 1),
    schedule="* * * * *",
    catchup=False,
    tags=["example"],
) as dag:
    load_data = BashOperator(
        task_id="load_data",
        bash_command=(
            "cd /Users/plinphon/HSclasses/M12_MLOps/mlops-starter-project-iris && "
            "python3 src/load_data.py"
        ),
    )

    split_dataset = BashOperator(
        task_id="split_dataset",
        bash_command=(
            "cd /Users/plinphon/HSclasses/M12_MLOps/mlops-starter-project-iris && "
            "python3 src/split_dataset.py"
        ),
    )
    test = BashOperator(task_id="test_echo", bash_command="echo Hello from Airflow")

    test >> load_data >> split_dataset
