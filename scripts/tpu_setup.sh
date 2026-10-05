#!/usr/bin/env bash
set -e

ZONE="${ZONE:-us-west4-a}"
TPU_NAME="${TPU_NAME:-mini-jev-tpu}"
PROJECT="${PROJECT:-gcp-ml-172005}"

echo "=== [1/3] TPU VM 연결 대기 중: ${TPU_NAME} (${ZONE}) ==="
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --command="echo 'TPU VM Connected!'"

echo "=== [2/3] TPU VM 내 JAX (TPU 가속) 및 의존성 설치 ==="
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --command="
    sudo apt-get update -y && sudo apt-get install -y python3-pip python3-venv
    python3 -m pip install --upgrade pip
    pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install flax safetensors transformers pydantic rich numpy huggingface_hub hf_transfer pillow soundfile
    pip install --upgrade jinja2
"

echo "=== [3/3] TPU 인식 검증 ==="
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --command="python3 -c 'import jax; print(\"Detected Devices:\", jax.devices())'"

echo "=== TPU 환경 세팅 완료 ==="
