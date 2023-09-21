from loguru import logger
from apps.api.models.base import DatasetStatusWebhookRequest
from tenacity import after_log, retry, stop_after_attempt, wait_fixed
from apps.api.utils.config import WEBHOOK_ENDPOINT
import requests


class WebhookHandler:
    def __init__(self) -> None:
        self.target_url = (
            WEBHOOK_ENDPOINT
            if WEBHOOK_ENDPOINT is not None
            else "https://build.withcontext.ai/api/webhook/chat"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        reraise=True,
        after=after_log(logger, 10),
    )
    def update_status(self, dataset_id: str, status: int):
        logger.info(f"Updating status of {dataset_id} to {status}")
        # add charcters and document's uid
        payload = DatasetStatusWebhookRequest(
            status=status, data={"api_dataset_id": dataset_id, "status": status}
        )
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.target_url, json=payload.dict(), headers=headers)
        response.raise_for_status()
