#!/usr/bin/env bash
set -e

ZONE="${ZONE:-us-west4-a}"
TPU_NAME="${TPU_NAME:-mini-jev-tpu}"
PROJECT="${PROJECT:-gcp-ml-172005}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== [1/2] 로컬 코드 및 오디오 데이터를 TPU VM으로 동기화 ==="
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --command="mkdir -p ~/mini-jev"

gcloud compute tpus tpu-vm scp --recurse \
  "${LOCAL_DIR}/mini_jev" \
  "${LOCAL_DIR}/benchmark_audio.py" \
  "${LOCAL_DIR}/data" \
  "${TPU_NAME}:~/mini-jev/" \
  --zone="${ZONE}" \
  --project="${PROJECT}"

echo "=== [2/2] TPU VM에서 Gemma 4 E2B 오디오 멀티모달 벤치마크 실행 ==="
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --command="export HF_TOKEN='${HF_TOKEN}' && cd ~/mini-jev && python3 benchmark_audio.py"
