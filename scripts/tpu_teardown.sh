#!/usr/bin/env bash
set -e

ZONE="${ZONE:-us-west4-a}"
TPU_NAME="${TPU_NAME:-mini-jev-tpu}"
QR_NAME="${QR_NAME:-mini-jev-qr-v5e4}"
PROJECT="${PROJECT:-gcp-ml-172005}"

echo "=== [1/2] TPU Queued Resource 삭제 중: ${QR_NAME} ==="
gcloud compute tpus queued-resources delete "${QR_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --force \
  --quiet || true

echo "=== [2/2] TPU VM 삭제 확인: ${TPU_NAME} ==="
gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --quiet || true

echo "=== TPU 자원 정리 완료 ==="
