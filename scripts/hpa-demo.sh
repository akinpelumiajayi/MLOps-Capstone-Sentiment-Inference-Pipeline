#!/usr/bin/env bash
# Drive enough load at the in-cluster Service to make the HPA scale out, and
# record replica count + CPU over time to reports/hpa-demo.log (use it, or a
# screen recording of this terminal, as evidence in the README).
#
# Load runs *inside* the cluster (Locust pod) so a laptop port-forward is not
# the bottleneck.
set -euo pipefail

NS="${1:-sentiment}"
USERS="${USERS:-150}"
DURATION="${DURATION:-5m}"
LOG="reports/hpa-demo.log"
mkdir -p reports

kubectl -n "$NS" create configmap locustfile --from-file=loadtest/locustfile.py \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NS" delete pod locust --ignore-not-found --wait
kubectl -n "$NS" run locust --image=locustio/locust:2.47.0 --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"lf","configMap":{"name":"locustfile"}}],
    "containers":[{"name":"locust","image":"locustio/locust:2.47.0",
      "args":["-f","/mnt/locustfile.py","--host","http://sentiment-api","--headless",
              "-u","'"$USERS"'","-r","20","-t","'"$DURATION"'","--only-summary"],
      "volumeMounts":[{"name":"lf","mountPath":"/mnt"}]}]}}'

echo "time,replicas,cpu_utilisation" | tee "$LOG"
end=$((SECONDS + 420))
while (( SECONDS < end )); do
  line=$(kubectl -n "$NS" get hpa sentiment-api \
    -o jsonpath='{.status.currentReplicas},{.status.currentMetrics[0].resource.current.averageUtilization}')
  echo "$(date +%H:%M:%S),$line" | tee -a "$LOG"
  phase=$(kubectl -n "$NS" get pod locust -o jsonpath='{.status.phase}' 2>/dev/null || echo Gone)
  [[ "$phase" == "Succeeded" || "$phase" == "Failed" ]] && break
  sleep 15
done

kubectl -n "$NS" logs locust | tail -25 | tee -a "$LOG"
kubectl -n "$NS" delete pod locust --ignore-not-found
echo "Saved $LOG. Scale-down follows after the 5-minute stabilisation window."
