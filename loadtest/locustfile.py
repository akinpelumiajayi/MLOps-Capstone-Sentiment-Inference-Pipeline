"""Load test for the sentiment API.

Interactive UI:      locust -f loadtest/locustfile.py --host http://localhost:8000
Headless (CI/demo):  locust -f loadtest/locustfile.py --host http://localhost:8000 \
                         --headless -u 50 -r 10 -t 2m --csv reports/loadtest

Traffic mix: 85% single predictions, 10% batches, 5% /info reads.
"""

from __future__ import annotations

import random

from locust import FastHttpUser, between, task

REVIEWS = [
    "A gripping, beautifully shot film with a career-best performance at its heart.",
    "Two hours of my life I will never get back. Dull, smug and badly paced.",
    "Funny, warm and surprisingly moving; the cast clearly had a great time.",
    "The plot is a mess and the dialogue is painfully clunky.",
    "Not perfect, but ambitious and often thrilling.",
    "It looks expensive, yet it feels completely hollow.",
    "an engrossing, intelligent drama that never talks down to its audience",
    "the jokes land rarely and the running time feels twice as long",
    "a modest but charming crowd-pleaser",
    "everything about this sequel is louder, dumber and less fun",
]


class SentimentUser(FastHttpUser):
    wait_time = between(0.05, 0.5)

    @task(17)
    def predict(self) -> None:
        self.client.post("/predict", json={"text": random.choice(REVIEWS)}, name="/predict")

    @task(2)
    def predict_batch(self) -> None:
        self.client.post("/predict/batch", json={"texts": random.sample(REVIEWS, k=8)}, name="/predict/batch")

    @task(1)
    def info(self) -> None:
        self.client.get("/info", name="/info")
