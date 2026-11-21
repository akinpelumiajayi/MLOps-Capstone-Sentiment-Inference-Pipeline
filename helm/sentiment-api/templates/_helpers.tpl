{{- define "sentiment-api.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Release "sentiment-api" -> resources named "sentiment-api" (not "sentiment-api-sentiment-api"). */}}
{{- define "sentiment-api.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "sentiment-api.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sentiment-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "sentiment-api.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{ include "sentiment-api.selectorLabels" . }}
app.kubernetes.io/version: {{ .Values.image.tag | default .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "sentiment-api.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "sentiment-api.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "sentiment-api.image" -}}
{{- $repo := required "image.repository is required" .Values.image.repository -}}
{{- $tag := required "image.tag is required (use an immutable tag, e.g. sha-<commit>)" .Values.image.tag -}}
{{- if eq $tag "latest" }}{{ fail "image.tag=latest is not allowed; deploy an immutable tag" }}{{ end -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end }}

{{/* True when the Prometheus Operator CRDs are installed in the cluster. */}}
{{- define "sentiment-api.hasPromOperator" -}}
{{- if .Capabilities.APIVersions.Has "monitoring.coreos.com/v1" }}true{{ end -}}
{{- end }}
