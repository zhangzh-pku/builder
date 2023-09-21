from celery import Celery
from apps.api.models.base.dataset import Dataset
from apps.api.models.controller import dataset_manager
from loguru import logger
from celery import current_task
from celery.exceptions import MaxRetriesExceededError
from urllib.parse import quote_plus
from utils.config import UPSTASH_REDIS_REST_TOKEN, UPSTASH_REDIS_REST_URL

UPSTASH_REDIS_REST_URL='usw2-secure-robin-30535.upstash.io'
UPSTASH_REDIS_REST_TOKEN='7ed79ff877ce4df7b03c8abdffc5a705'
app = Celery('tasks')

# Configuration

broker_host = UPSTASH_REDIS_REST_URL
broker_port = 30535
broker_db = 0  # Database 1 for the broker
results_db = 0  # Database 2 for the results
password = UPSTASH_REDIS_REST_TOKEN

app.conf.broker_url = f"rediss://:{password}@{broker_host}:{broker_port}/{broker_db}?ssl_cert_reqs=CERT_REQUIRED"
app.conf.result_backend = f"rediss://:{password}@{broker_host}:{broker_port}/{results_db}?ssl_cert_reqs=CERT_REQUIRED"



@app.task(bind=True)
def background_upsert_dataset(self, id: str, dataset_info: dict):
    try:
        dataset_manager.upsert_dataset(id, dataset_info)
        logger.info(f"Upsert for dataset {id} completed.")
        self.update_state(state='PROGRESS', meta={'progress': 50})
    except Exception as e:
        logger.error(f"Error during upsert for dataset {id}: {e}")
        try:
            # retry the task in 60 seconds
            self.retry(countdown=60)
        except MaxRetriesExceededError:
            pass


@app.task(bind=True)
def background_create_dataset(self, dataset: Dataset):
    try:
        dataset_manager.save_dataset(dataset)
        logger.info(f"Dataset {dataset.id} created.")
        self.update_state(state='PROGRESS', meta={'progress': 50})
    except Exception as e:
        logger.error(f"Error during creation of dataset {dataset.id}: {e}")
        try:
            # retry the task in 60 seconds
            self.retry(countdown=60)
        except MaxRetriesExceededError:
            pass
