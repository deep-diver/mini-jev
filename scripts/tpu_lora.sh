#!/usr/bin/env bash
# Provision a single-chip TPU and run the LoRA + pointer head training on it.
#
# train_lora_tpu.py is the trainer; this is everything around it. The weights
# are pushed from the local Hugging Face cache rather than downloaded on the
# VM, so the TPU never needs a token for the gated repo.
#
#   scripts/tpu_lora.sh create          # ~5 min
#   scripts/tpu_lora.sh setup           # jax[tpu] and friends
#   scripts/tpu_lora.sh push            # weights, data, scripts
#   scripts/tpu_lora.sh train
#   scripts/tpu_lora.sh eval
#   scripts/tpu_lora.sh fetch           # checkpoint back to data/decision/
#   scripts/tpu_lora.sh stop            # billing stops; disk is kept
#   scripts/tpu_lora.sh delete          # nothing is kept
#
# Measured: 0.082 s/step at batch 32 on v5e-1, so 2000 steps is about 3 minutes.
set -euo pipefail

ZONE="${ZONE:-us-west4-a}"
TPU_NAME="${TPU_NAME:-jev-tpu}"
ACCELERATOR="${ACCELERATOR:-v5litepod-1}"
VERSION="${VERSION:-v2-alpha-tpuv5-lite}"
MODEL="${MODEL:-google/gemma-3-270m-it}"
REMOTE_SUBDIR="${REMOTE_SUBDIR:-jev}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Capacity is per zone and moves around; v5e-1 quota is TPU_LITE_PODSLICE_V5.
FALLBACK_ZONES="${FALLBACK_ZONES:-us-west4-a us-east1-c us-central1-a europe-west4-a}"

ssh_() { gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --command="$1"; }
scp_() { gcloud compute tpus tpu-vm scp "$1" "$TPU_NAME:$2" --zone="$ZONE" >/dev/null; }

# scp does not expand ~ or $HOME in the destination, so resolve it once and
# use absolute paths everywhere.
remote_dir() {
  if [ -z "${REMOTE:-}" ]; then
    REMOTE="$(ssh_ 'echo $HOME' 2>/dev/null | tr -d "\r" | grep "^/" | tail -1)/$REMOTE_SUBDIR"
    [ "$REMOTE" = "/$REMOTE_SUBDIR" ] && { echo "원격 홈을 찾지 못했습니다"; exit 1; }
  fi
  echo "$REMOTE"
}

case "${1:-}" in

create)
  for z in $ZONE $FALLBACK_ZONES; do
    echo "=== $z"
    if gcloud compute tpus tpu-vm create "$TPU_NAME" --zone="$z" \
         --accelerator-type="$ACCELERATOR" --version="$VERSION" 2>&1 | tail -2; then
      if gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$z" \
           --format="value(state)" 2>/dev/null | grep -q .; then
        echo "생성됨: $z   (이후 명령에 ZONE=$z 를 쓰세요)"
        exit 0
      fi
    fi
  done
  echo "모든 존에서 실패"; exit 1
  ;;

setup)
  # transformers 는 토크나이저만 쓰므로 --no-deps 로 받아 torch 를 끌어오지 않는다.
  # optax 가 조용히 요구하는 absl-py / chex / etils / toolz 를 함께 넣는다.
  ssh_ '
    set -e
    pip install -q -U pip
    pip install -q "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install -q optax safetensors "transformers>=4.44" --no-deps
    pip install -q tokenizers huggingface_hub regex filelock pyyaml tqdm packaging requests
    pip install -q absl-py chex etils toolz
    python3 -c "
import jax, optax
print(\"jax\", jax.__version__, jax.default_backend(), jax.devices())
from transformers import AutoTokenizer; import safetensors.flax
print(\"deps OK\")"'
  ;;

push)
  REMOTE="$(remote_dir)"
  snap=$(python3 - "$MODEL" <<'PY'
import os, sys
from huggingface_hub import try_to_load_from_cache
p = try_to_load_from_cache(sys.argv[1], "config.json")
if not isinstance(p, str):
    sys.exit("모델이 로컬 캐시에 없습니다: " + sys.argv[1])
print(os.path.dirname(p))
PY
)
  echo "snapshot: $snap"
  # 스냅샷은 blobs 로 가는 심볼릭 링크라 -h 로 실체를 담는다.
  rm -rf /tmp/jev_model && mkdir -p /tmp/jev_model
  cp -L "$snap"/* /tmp/jev_model/
  tar -czf /tmp/jev_model.tgz -C /tmp/jev_model .
  echo "가중치 $(du -h /tmp/jev_model.tgz | cut -f1) 전송"

  ssh_ "mkdir -p $REMOTE/data"
  scp_ /tmp/jev_model.tgz "$REMOTE/"
  for f in scripts/train_lora_tpu.py scripts/eval_lora.py; do scp_ "$ROOT/$f" "$REMOTE/"; done
  for f in data/decision/train.jsonl data/decision/dev.jsonl; do scp_ "$ROOT/$f" "$REMOTE/data/"; done
  scp_ "$ROOT/data/eval/alphaxiv_multi.json" "$REMOTE/"
  ssh_ "cd $REMOTE && rm -rf gemma_model && mkdir gemma_model &&
        tar -xzf jev_model.tgz -C gemma_model && rm jev_model.tgz &&
        ls gemma_model | tr '\n' ' ' && echo && wc -l data/*.jsonl"
  ;;

train)
  REMOTE="$(remote_dir)"
  shift || true
  ssh_ "cd $REMOTE && nohup python3 train_lora_tpu.py --model $REMOTE/gemma_model \
        --data $REMOTE/data --out $REMOTE/lora.npz \
        --steps ${STEPS:-2000} --eval-every 250 --batch ${BATCH:-32} \
        --max-len 256 --rank 16 --dim 256 --lr 1e-3 $* > $REMOTE/train.log 2>&1 &
        echo started"
  echo "진행 상황: ZONE=$ZONE $0 log"
  ;;

log)
  REMOTE="$(remote_dir)"
  ssh_ "pgrep -f train_lora_tpu >/dev/null && echo running || echo finished;
        grep -v 'PyTorch was not found' $REMOTE/train.log | tail -20"
  ;;

eval)
  REMOTE="$(remote_dir)"
  ssh_ "cd $REMOTE && python3 eval_lora.py --model $REMOTE/gemma_model \
        --ckpt $REMOTE/lora.npz --data $REMOTE/alphaxiv_multi.json 2>&1 \
        | grep -v 'PyTorch was not found'"
  ;;

fetch)
  REMOTE="$(remote_dir)"
  out="${2:-$ROOT/data/decision/lora_tpu.npz}"
  gcloud compute tpus tpu-vm scp "$TPU_NAME:$REMOTE/lora.npz" "$out" --zone="$ZONE"
  ls -lh "$out"
  ;;

stop)   gcloud compute tpus tpu-vm stop   "$TPU_NAME" --zone="$ZONE" | tail -1 ;;
start)  gcloud compute tpus tpu-vm start  "$TPU_NAME" --zone="$ZONE" | tail -1 ;;
delete) gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --quiet ;;

*)
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
  ;;
esac
