# Standard
import json

# Third Party
from locust import FastHttpUser, task


class QuickstartUser(FastHttpUser):
    #  wait_time = between(1, 5)

    @task
    def forecast_synchronous(self):
        forecasting_url = self.host + "/inference/forecasting"

        if forecasting_url.find("fmaas") >= 0:
            input.pop("s3credentials")

        self.client.post(forecasting_url, json=self.payload, timeout=None, retries=10)

    def on_start(self):
        self.payload = json.load(open("./payload.json"))
