# MLOps Capstone — Sentiment Inference Pipeline

A production-grade MLOps pipeline that covers the full lifecycle of an ML model:
training, packaging, CI/CD, containerisation, Kubernetes deployment, autoscaling,
and observability.

**Stack:** FastAPI · sklearn · DVC · Docker · GitHub Actions · Jenkins · Helm · Kubernetes (HPA) · Prometheus · Grafana · Streamlit

---

## Quick Start (local Docker)

```bash
# 1. Train the model
docker compose --profile train up trainer
# or: python training/train.py

# 2. Start the full stack
docker compose up --build

# 3. Open
#   API docs    → http://localhost:8000/docs
#   Dashboard   → http://localhost:8501
#   Prometheus  → http://localhost:9090
#   Grafana     → http://localhost:3000  (admin / admin123)
```

---

## DVC Workflow

```bash
pip install -r requirements-train.txt

# Configure S3 remote (set your bucket in .dvc/config first)
dvc remote modify myremote access_key_id     $AWS_ACCESS_KEY_ID
dvc remote modify myremote secret_access_key $AWS_SECRET_ACCESS_KEY

# Run full pipeline
dvc repro

# View metrics
dvc metrics show

# Push model + data to remote
dvc push
```

---

## Kubernetes Deployment

### Local cluster (kind)

```bash
# Create cluster with metrics-server
kind create cluster --name mlops

# Install metrics-server (required for HPA)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system \
  --type='json' -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

# Load local image
docker build -t sentiment-api:local .
kind load docker-image sentiment-api:local --name mlops

# Deploy
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/

# Port-forward API
kubectl port-forward svc/sentiment-api 8000:80 -n sentiment-prod
```

### Helm deploy (recommended)

```bash
# Install kube-prometheus-stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kube-prom prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set prometheusOperator.admissionWebhooks.enabled=false \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# Deploy app
helm upgrade --install sentiment-api ./helm/mlops-chart \
  --namespace sentiment-prod \
  --create-namespace \
  --set image.repository=ghcr.io/YOUR_ORG/sentiment-api \
  --set image.tag=latest

# Watch HPA
kubectl get hpa -n sentiment-prod -w
```

---

## CI/CD

| Platform | File |
|---|---|
| GitHub Actions | `.github/workflows/ci-cd.yml` |
| Jenkins | `jenkins/Jenkinsfile` |

**Required GitHub Secrets:**

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | DVC S3 remote access |
| `AWS_SECRET_ACCESS_KEY` | DVC S3 remote secret |
| `KUBECONFIG` | Base64-encoded kubeconfig for deploy target |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | `{"text": "..."}` → `{label, confidence, model_version}` |
| `/healthz` | GET | Liveness check |
| `/metrics` | GET | Prometheus scrape endpoint |
| `/docs` | GET | Interactive Swagger UI |

---

## Project Structure

```
mlops-capstone/
├── app/
│   ├── main.py                        # FastAPI inference server
│   └── model.py                       # sklearn TF-IDF + LogReg pipeline
├── training/
│   └── train.py                       # DVC-tracked training stage
├── streamlit_app/
│   ├── app.py                         # 3-tab monitoring dashboard
│   └── Dockerfile.streamlit
├── k8s/                               # Raw Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   └── servicemonitor.yaml
├── helm/
│   └── mlops-chart/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── _helpers.tpl
│           ├── configmap.yaml
│           ├── deployment.yaml
│           ├── service.yaml
│           ├── hpa.yaml
│           └── servicemonitor.yaml
├── .github/
│   └── workflows/
│       └── ci-cd.yml                  # 4-stage GitHub Actions pipeline
├── jenkins/
│   └── Jenkinsfile                    # 5-stage declarative pipeline
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── provisioning/
│           ├── datasources/prometheus.yaml
│           └── dashboards/dashboard.yaml
├── tests/
│   └── test_api.py                    # pytest integration tests
├── data/
│   ├── raw/.gitkeep                   # DVC-tracked, ignored by git
│   └── processed/.gitkeep
├── models/
│   └── .gitkeep                       # model.pkl managed by DVC
├── .dvc/
│   └── config                         # DVC S3 remote config
├── Dockerfile                         # API image
├── Dockerfile.train                   # Training job image
├── docker-compose.yml                 # Full local stack
├── params.yaml                        # DVC hyperparameters
├── dvc.yaml                           # DVC pipeline definition
├── requirements.txt                   # API dependencies
├── requirements-train.txt             # Training dependencies
├── .gitignore
└── mlops-capstone.code-workspace      # VSCode workspace
```

---

## Grafana Dashboards (PromQL)

```promql
# Requests per second
rate(sentiment_requests_total[1m])

# p95 latency
histogram_quantile(0.95, rate(sentiment_request_duration_seconds_bucket[5m]))

# Confidence distribution
histogram_quantile(0.50, rate(sentiment_confidence_score_bucket[5m]))

# HPA replica count
kube_horizontalpodautoscaler_status_current_replicas{
  horizontalpodautoscaler="sentiment-api",
  namespace="sentiment-prod"
}
```

---

## Key Gotchas

- **ServiceMonitor label** — `release: kube-prom` in `servicemonitor.yaml` must exactly match your Prometheus Helm release name
- **kind image loading** — use `imagePullPolicy: Never` and `kind load docker-image` for local images
- **metrics-server** — required for HPA CPU metrics; patch with `--kubelet-insecure-tls` for kind
- **DVC remote** — set your S3 bucket in `.dvc/config` before running `dvc push`/`dvc pull`

---

*Sheffield Hallam University — AI R&D Capstone*
