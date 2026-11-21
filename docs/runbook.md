# Runbook

One section per alert in [`helm/sentiment-api/files/alerts.yml`](../helm/sentiment-api/files/alerts.yml).
Each section covers what the alert means, how to confirm it, and what to do.

Useful commands (namespace `sentiment-prod`):

```bash
kubectl -n sentiment-prod get pods,hpa,pdb
kubectl -n sentiment-prod logs -l app.kubernetes.io/name=sentiment-api --tail=200 | jq .
kubectl -n sentiment-prod port-forward svc/sentiment-api 8000:80 && curl localhost:8000/info
helm -n sentiment-prod history sentiment-api
helm -n sentiment-prod rollback sentiment-api <REVISION>   # restores previous image = previous model
```

Every image tag is `sha-<commit>` and carries exactly one model, so **rolling back
the release rolls back the model**. No separate model store has to be kept in step.

---

## SentimentAPIDown

**Meaning:** Prometheus cannot scrape a pod for 2 minutes.
**Check:** `kubectl get pods`, then look for `CrashLoopBackOff` or `OOMKilled` (`kubectl describe pod`).
**Act:** For OOM, raise `resources.limits.memory`. For crash loops, read the logs. If a new
release caused it, `helm rollback`.

## SentimentModelNotLoaded

**Meaning:** The process is alive (liveness passes) but `/readyz` fails because the model did
not load. Kubernetes keeps these pods out of the Service, so users are unaffected while healthy
replicas remain.
**Check:** Logs contain `"msg": "Model failed to load"` with the traceback.
**Act:** Usually a corrupt or incompatible artifact, such as a scikit-learn version mismatch
between training and serving. Roll back, then fix the pins in `requirements*.txt`.

## SentimentHighErrorRate

**Meaning:** More than 2% of requests returned 5xx over 5 minutes.
**Check:** Grafana, panel "Requests by route and status". Find failing requests in the logs by
`status` and `request_id`.
**Act:** If it started with a deploy, roll back. 4xx responses (bad client input) do not count
towards this alert.

## SentimentHighP95Latency

**Meaning:** p95 `/predict` latency is above 100 ms for 10 minutes. Model inference alone is
about 1 ms (see `reports/metrics.json`), so the time is going elsewhere.
**Check:** Compare "HTTP latency" against "Model inference time". If HTTP latency is high but
inference is not, the pods are CPU-saturated or queueing. Check HPA replicas against
`maxReplicas`.
**Act:** Raise `hpa.maxReplicas` or the CPU requests. Check for very long inputs ("Input length"
panel).

## SentimentPredictionDrift

**Meaning:** The distribution of P(positive) over the last 500 predictions per pod differs from
the test-set reference with PSI > 0.2.
**Check:** Grafana "Score PSI". Is "input length PSI" also high? Did a new client or traffic
source start recently? Export a sample of recent inputs and run
`python -m sentiment.drift --input sample.csv` for an offline confirmation.
**Act:** Drift is a signal to investigate, not to roll back. If the inputs are a new domain
(for example product reviews instead of film reviews), label a sample, add it to the training
data and retrain through the normal PR flow. The quality gate then compares the new model with
the current one.

## SentimentPositiveRateShift

**Meaning:** The share of positive predictions moved more than 15 points from the reference rate
(about 50%).
**Check:** Is the shift real (for example a genuinely bad release being reviewed) or caused by
the model? Spot-check recent predictions by hand.
**Act:** As for drift. Escalate to the data/ML owner if labels are needed.

## SentimentHighOOVRate

**Meaning:** More than 40% of input tokens are unknown to the TF-IDF vocabulary, so predictions
rest on very little evidence.
**Check:** "Mean out-of-vocabulary ratio" panel. Look at sample inputs: is the language
different, or is the text mostly emojis, URLs or code?
**Act:** Add preprocessing for the new input type, or retrain with in-domain data.

## SentimentLowConfidenceSpike

**Meaning:** More than half of predictions fall below `LOW_CONFIDENCE_THRESHOLD`.
**Check:** It often appears together with drift or a high OOV rate.
**Act:** Downstream consumers should route `low_confidence: true` results to human review.
Investigate as for drift.
