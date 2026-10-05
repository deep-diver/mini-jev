#!/usr/bin/env bash
set -e

ZONE="${ZONE:-us-west4-a}"
TPU_NAME="${TPU_NAME:-mini-jev-tpu}"
QR_NAME="${QR_NAME:-mini-jev-qr-v5e4}"
PROJECT="${PROJECT:-gcp-ml-172005}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v5litepod-4}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv5-lite}"

# Gemma is a gated model, so the TPU VM needs a Hugging Face token. Prefer the
# environment, then fall back to whatever `huggingface-cli login` already stored
# on this machine -- there is then nothing extra to export before running this.
resolve_hf_token() {
  if [ -n "${HF_TOKEN}" ]; then
    return 0
  fi
  HF_TOKEN="$(python3 - <<'EOF' 2>/dev/null
try:
    from huggingface_hub import get_token
    tok = get_token()
except ImportError:
    from huggingface_hub import HfFolder
    tok = HfFolder.get_token()
print(tok or "")
EOF
)"
  export HF_TOKEN
}

require_hf_token() {
  resolve_hf_token
  if [ -z "${HF_TOKEN}" ]; then
    echo "ERROR: no Hugging Face token found."
    echo "  google/gemma-* are gated, so the TPU VM cannot download them without one."
    echo "  Fix with:  huggingface-cli login       (or: export HF_TOKEN=...)"
    exit 1
  fi
  echo "Hugging Face token: found (not printed)"
}

# Check before creating anything: billing starts the moment the TPU is ACTIVE,
# and a missing token would only surface later, after the model download fails.
require_hf_token

echo "=== [1/2] Cloud TPU v5e Spot Queued Resource 생성 요청: ${QR_NAME} ==="
gcloud compute tpus queued-resources create "${QR_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --accelerator-type="${ACCELERATOR_TYPE}" \
  --runtime-version="${RUNTIME_VERSION}" \
  --node-id="${TPU_NAME}" \
  --spot \
  --quiet

echo "=== [2/2] TPU 리소스 ACTIVE 상태 대기 중... ==="
while true; do
  STATE=$(gcloud compute tpus queued-resources describe "${QR_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --format="value(state.state)")
  echo "Current Queued Resource State: ${STATE}"
  if [ "${STATE}" == "ACTIVE" ]; then
    echo "TPU is ACTIVE and READY!"
    break
  fi
  if [ "${STATE}" == "FAILED" ] || [ "${STATE}" == "SUSPENDED" ]; then
    echo "Error: TPU creation entered unexpected state: ${STATE}"
    exit 1
  fi
  sleep 10
done
