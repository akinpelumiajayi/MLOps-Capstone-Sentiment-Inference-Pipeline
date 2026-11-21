#!/usr/bin/env bash
# Build the shippable image, side-load it into kind, deploy with Helm, smoke test.
set -euo pipefail

CLUSTER="${1:-mlops}"
NS="${2:-sentiment}"
IMAGE="sentiment-api:local"

for f in models/model.pkl models/metadata.json models/reference_profile.json; do
  [[ -f "$f" ]] || { echo "Missing $f: run 'make train' first." >&2; exit 1; }
done

docker build --target runtime -t "$IMAGE" .
kind load docker-image "$IMAGE" --name "$CLUSTER"

helm upgrade --install sentiment-api helm/sentiment-api \
  -f helm/sentiment-api/values-kind.yaml \
  --namespace "$NS" --create-namespace --wait --timeout 3m
# Same tag, new image: force pods to pick up the freshly loaded build.
kubectl -n "$NS" rollout restart deployment/sentiment-api
kubectl -n "$NS" rollout status deployment/sentiment-api --timeout=120s
kubectl -n "$NS" get pods,svc,hpa,pdb

kubectl -n "$NS" port-forward svc/sentiment-api 8000:80 >/dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
python scripts/smoke_test.py http://localhost:8000
echo "Deployed. Port-forward with: kubectl -n $NS port-forward svc/sentiment-api 8000:80"
