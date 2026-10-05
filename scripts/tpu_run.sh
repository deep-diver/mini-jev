#!/usr/bin/env bash
set -e

ZONE="${ZONE:-us-west4-a}"
TPU_NAME="${TPU_NAME:-mini-jev-tpu}"
PROJECT="${PROJECT:-gcp-ml-172005}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

require_hf_token

echo "=== [1/3] 로컬 mini-jev 코드를 TPU VM으로 동기화 ==="
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --command="mkdir -p ~/mini-jev"

gcloud compute tpus tpu-vm scp --recurse \
  "${LOCAL_DIR}/mini_jev" \
  "${LOCAL_DIR}/examples" \
  "${LOCAL_DIR}/scripts" \
  "${LOCAL_DIR}/benchmark.py" \
  "${LOCAL_DIR}/benchmark_audio.py" \
  "${LOCAL_DIR}/data" \
  "${TPU_NAME}:~/mini-jev/" \
  --zone="${ZONE}" \
  --project="${PROJECT}"

echo "=== [2/4] TPU VM에서 기본 데모 실행 (examples/basic_usage.py) ==="
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --command="export HF_TOKEN='${HF_TOKEN}' && cd ~/mini-jev && python3 examples/basic_usage.py"

echo "=== [3/4] TPU VM에서 4-Way 멀티 디바이스 병렬 데모 실행 (examples/multi_device_parallel.py) ==="
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --command="export HF_TOKEN='${HF_TOKEN}' && cd ~/mini-jev && python3 examples/multi_device_parallel.py"

echo "=== [4/5] TPU VM에서 멀티 디바이스 벤치마크 실행 (benchmark.py --parallel) ==="
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --command="export HF_TOKEN='${HF_TOKEN}' && cd ~/mini-jev && python3 benchmark.py --parallel"

echo "=== [5/6] TPU VM에서 SurfMate 결정 태스크 벤치마크 (Gemma 3 270M) ==="
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --command="export HF_TOKEN='${HF_TOKEN}' && cd ~/mini-jev && python3 scripts/tpu_bench_surfmate.py --model google/gemma-3-270m-it"

echo "=== [6/6] TPU VM에서 SurfMate 결정 태스크 벤치마크 (Gemma 4 E2B) ==="
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --command="export HF_TOKEN='${HF_TOKEN}' && cd ~/mini-jev && python3 scripts/tpu_bench_surfmate.py --model google/gemma-4-E2B-it"

echo "=== [보조] 오디오 멀티모달 벤치마크 (benchmark_audio.py) ==="
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT}" \
  --command="export HF_TOKEN='${HF_TOKEN}' && cd ~/mini-jev && python3 benchmark_audio.py"
