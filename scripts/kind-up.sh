#!/usr/bin/env bash
# Create a local kind cluster with metrics-server (required for the HPA) and,
# optionally, kube-prometheus-stack (WITH_MONITORING=1).
set -euo pipefail

CLUSTER="${1:-mlops}"
METRICS_SERVER_VERSION="v0.8.0"

if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER" --wait 120s
fi

kubectl apply -f "https://github.com/kubernetes-sigs/metrics-server/releases/download/${METRICS_SERVER_VERSION}/components.yaml"
# kind's kubelets use self-signed certs.
kubectl -n kube-system patch deployment metrics-server --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' || true
kubectl -n kube-system rollout status deployment/metrics-server --timeout=120s

if [[ "${WITH_MONITORING:-0}" == "1" ]]; then
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
  helm repo update >/dev/null
  # Release name must match monitoring.releaseLabel in values.yaml (kube-prom).
  helm upgrade --install kube-prom prometheus-community/kube-prometheus-stack \
    --namespace monitoring --create-namespace --wait --timeout 10m \
    --set grafana.adminPassword="${GRAFANA_ADMIN_PASSWORD:-change-me-please}"
  kubectl -n monitoring create configmap sentiment-dashboard \
    --from-file=monitoring/grafana/dashboards/sentiment-api.json --dry-run=client -o yaml \
    | kubectl label --local -f - grafana_dashboard=1 -o yaml | kubectl apply -f -
fi

echo "Cluster '$CLUSTER' ready."
