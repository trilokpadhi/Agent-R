#!/usr/bin/env bash
# Agent-R reproduction pipeline, without Kubernetes.
#
#   scripts/run_agentr.sh pipeline/config.yaml
#
# Runs on one node with N GPUs (Slurm, PBS, or a bare machine). It performs exactly the steps the
# Kubernetes controller performs, reading the SAME config file, so a run here and a run there differ
# only in how the work is placed on GPUs:
#
#   for each iteration:
#     1. search      MCTS trees                (mcts_collection.py, 1 GPU per shard)
#     2. revise      revision trajectories     (path_collection.py, 1 GPU per shard)
#     3. sft-data    build train.jsonl         (controller.py --step sft-data, CPU)
#     4. sft         full fine-tune            (swift sft, all GPUs)
#     5. eval        score the checkpoint      (eval.py, 1 GPU per shard)
#
# Resuming: rerun the same command. Each step writes .done when it finishes and is then skipped,
# and within a step each task's tree / each eval item is skipped if its output file already exists.
# So an interrupted run (a Slurm time limit, a crash) resumes roughly where it stopped.
#
# Site paths - model, conda envs, environment data - come from scripts/site.env. Copy
# scripts/site.env.example and edit it. Everything about the METHOD comes from the config file;
# nothing that affects results is set here.
set -uo pipefail
[ "${BASH_VERSINFO[0]:-0}" -ge 4 ] || { echo "needs bash 4+ (this is ${BASH_VERSION:-unknown});"\
  " on macOS: brew install bash" >&2; exit 1; }

CONFIG=${1:?usage: scripts/run_agentr.sh pipeline/<config>.yaml}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SITE=${AGENTR_SITE_ENV:-$REPO/scripts/site.env}
[ -f "$SITE" ] || { echo "missing $SITE - copy scripts/site.env.example and edit it" >&2; exit 1; }
# shellcheck disable=SC1090
source "$SITE"
eval "$(python3 "$REPO/scripts/config_env.py" "$CONFIG")" || exit 1

RUN_DIR=$RUN_ROOT/$RUN
LOG_DIR=$RUN_DIR/logs
mkdir -p "$LOG_DIR"

# Progress goes to stderr: step_sft_data and step_sft return paths on stdout.
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >&2; }
die() { log "PIPELINE_FAILED: $*"; exit 1; }
done_file() { [ -f "$1/.done" ]; }
mark_done() { mkdir -p "$1"; printf '%s\n' "$2" > "$1/.done"; }

# Index a space-separated list (the config's per-iteration alpha and epochs lists) from 1.
nth() { local i=$1; shift; local a=("$@"); echo "${a[$((i - 1))]}"; }

# ---------------------------------------------------------------- backends
# One vLLM server and one environment server per GPU, mirroring one Kubernetes pod per GPU.
# Ports: vLLM 8000+gpu, environment 36001+gpu.
BACKEND_PIDS=()

vllm_port() { echo $((8000 + $1)); }
env_port()  { echo $((36001 + $1)); }

start_vllm() {
  local gpu=$1 model=$2 port
  port=$(vllm_port "$gpu")
  CUDA_VISIBLE_DEVICES=$gpu HF_HOME=$AGENTR_HF_HOME HF_HUB_OFFLINE=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "$AGENTR_POLICY_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$model" --dtype "$VLLM_DTYPE" --gpu-memory-utilization 0.90 --port "$port" \
    > "$LOG_DIR/vllm.gpu$gpu.log" 2>&1 &
  BACKEND_PIDS+=($!)
}

start_env_server() {
  local task=$1 gpu=$2 port
  port=$(env_port "$gpu")
  local proto_var="${task^^}_PROTOCOL"
  case "$task:${!proto_var}" in
    webshop:eto)
      ( export PYTHONPATH=$AGENTR_WEBSHOP_ETO_ROOT/envs/webshop/src HOST=127.0.0.1 PORT=$port
        cd "$AGENTR_WEBSHOP_ETO_ROOT" || exit 1
        exec "$AGENTR_ENV_WEBSHOP_PYTHON" "$REPO/webshop_eto/server.py" ) \
        > "$LOG_DIR/env.$task.gpu$gpu.log" 2>&1 & ;;
    webshop:*)
      ( cd "$AGENTR_AGENTGYM_ROOT/agentenv-webshop/webshop" || exit 1
        exec "$AGENTR_ENV_WEBSHOP_BIN" --host 127.0.0.1 --port "$port" ) \
        > "$LOG_DIR/env.$task.gpu$gpu.log" 2>&1 & ;;
    sciworld:eto)
      ( export PYTHONPATH=$AGENTR_SCIWORLD_ETO_ROOT:$REPO HOST=127.0.0.1 PORT=$port
        export SCIWORLD_ETO_ROOT=$AGENTR_SCIWORLD_ETO_ROOT SCIWORLD_SPLIT=${SCIWORLD_SPLIT:-train}
        cd "$AGENTR_SCIWORLD_ETO_ROOT" || exit 1
        exec "$AGENTR_ENV_SCIWORLD_PYTHON" "$REPO/sciworld_eto/server.py" ) \
        > "$LOG_DIR/env.$task.gpu$gpu.log" 2>&1 & ;;
    sciworld:*)
      ( cd "$AGENTR_AGENTGYM_AGENTR_ROOT/agentenv-sciworld" || exit 1
        exec "$AGENTR_ENV_SCIWORLD_PYTHON" -c \
          "import uvicorn; uvicorn.run('agentenv_sciworld:app', host='127.0.0.1', port=$port)" ) \
        > "$LOG_DIR/env.$task.gpu$gpu.log" 2>&1 & ;;
  esac
  BACKEND_PIDS+=($!)
}

wait_http() {  # wait_http <url> <seconds> <what>
  local url=$1 limit=$2 what=$3 i
  for ((i = 0; i < limit; i += 5)); do
    "$AGENTR_POLICY_PYTHON" -c "import urllib.request,sys; urllib.request.urlopen('$url', timeout=3)" \
      2>/dev/null && return 0
    sleep 5
  done
  log "TIMEOUT waiting for $what at $url after ${limit}s"
  return 1
}

stop_backends() {
  local pid
  # vLLM forks an EngineCore process and the env servers fork workers, so kill children first;
  # a bare kill of the parent would leave those holding GPU memory and the ports.
  for pid in "${BACKEND_PIDS[@]:-}"; do [ -n "$pid" ] && pkill -P "$pid" 2>/dev/null; done
  for pid in "${BACKEND_PIDS[@]:-}"; do [ -n "$pid" ] && kill "$pid" 2>/dev/null; done
  sleep 3
  BACKEND_PIDS=()
  wait 2>/dev/null
}
trap stop_backends EXIT

start_backends() {  # start_backends <task> <model> [with_env_server]
  local task=$1 model=$2 with_env=${3:-yes} gpu
  for ((gpu = 0; gpu < GPUS; gpu++)); do
    start_vllm "$gpu" "$model"
    [ "$with_env" = yes ] && start_env_server "$task" "$gpu"
  done
  for ((gpu = 0; gpu < GPUS; gpu++)); do
    wait_http "http://127.0.0.1:$(vllm_port "$gpu")/health" 1800 "vLLM on GPU $gpu" || return 1
    if [ "$with_env" = yes ]; then
      # The full WebShop catalogue takes ~75s per process; ScienceWorld starts a JVM per session.
      wait_http "http://127.0.0.1:$(env_port "$gpu")/" 1800 "$task server on GPU $gpu" || return 1
    fi
  done
  log "backends ready: $GPUS vLLM servers$([ "$with_env" = yes ] && echo " + $GPUS $task servers")"
}

# Environment shared by mcts_collection.py, path_collection.py and eval.py.
export_inference_env() {  # export_inference_env <task> <model_dir> <model_type> <temp>
  local task=$1
  export TASK=$task MODEL_DIR=$2 MODEL_TYPE=$3 TEMP=$4
  export MODEL_NAME MAX_DEPTH ITERA N_GEN MAX_TOKEN_LENGTH MAX_NEW_TOKENS
  export VLLM_DTYPE MCTS_BATCH_GEN MCTS_PROFILE STOP_TOKENS=""
  # Only exported when the config sets it; an absent variable means "do not pass
  # chat_template_kwargs at all", which is what a non-Qwen template needs.
  [ -n "${ENABLE_THINKING:-}" ] && export ENABLE_THINKING
  export WEBSHOP_PROTOCOL SCIWORLD_PROTOCOL INTERCODE_PROTOCOL
  export HF_HOME=$AGENTR_HF_HOME HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  local agentenv_var="AGENTENV_${task^^}"
  export PYTHONPATH="$REPO:${!agentenv_var}:$AGENTR_POLICY_SITE"
  # The eto clients read their prompt and split files at IMPORT time, from these roots. On the
  # Kubernetes cluster the built-in defaults (/data/src/...) happen to be correct, so nothing sets
  # them; anywhere else the policy side dies with FileNotFoundError on sciworld_inst.txt before it
  # runs a single task. Export whichever the site defines.
  [ -n "${AGENTR_WEBSHOP_ETO_ROOT:-}" ]   && export WEBSHOP_ETO_ROOT="$AGENTR_WEBSHOP_ETO_ROOT"
  [ -n "${AGENTR_SCIWORLD_ETO_ROOT:-}" ]  && export SCIWORLD_ETO_ROOT="$AGENTR_SCIWORLD_ETO_ROOT"
  [ -n "${AGENTR_INTERCODE_ETO_ROOT:-}" ] && export INTERCODE_ETO_ROOT="$AGENTR_INTERCODE_ETO_ROOT"
  return 0
}

latest_checkpoint() {  # highest checkpoint-<n> under $1, by number
  python3 - "$1" <<'EOF'
import sys, pathlib
d = pathlib.Path(sys.argv[1])
ck = [p for p in d.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()]
print(max(ck, key=lambda p: int(p.name.split("-")[-1])) if ck else "")
EOF
}

# ---------------------------------------------------------------- 1. search
step_search() {
  local it=$1 task=$2 model=$3
  local step_dir=$RUN_DIR/iter$it/search-$task
  done_file "$step_dir" && { log "iter$it search $task: done, skipping"; return 0; }
  local trees=$step_dir/mcts_result/$task/$MODEL_NAME
  mkdir -p "$trees"

  local shards_var="SHARDS_${task^^}" expected_var="EXPECTED_${task^^}"
  local shards=(${!shards_var}) expected=${!expected_var}
  log "iter$it search $task: ${#shards[@]} shards ${shards[*]}, model $model"

  export SCIWORLD_SPLIT=train   # collection is always on the training split
  export_inference_env "$task" "$model" Raw "$TEMP"
  start_backends "$task" "$model" || die "search $task: backends did not start"
  ( cd "$step_dir" && ln -sfn "$REPO/mcts_utils" mcts_utils )

  local pids=() gpu=0 spec lo hi w span wlo whi extra=""
  [ "$task" = sciworld ] && extra="--task_iteration $SCIWORLD_TASK_ITERATION"
  for spec in "${shards[@]}"; do
    lo=${spec%%:*}; hi=${spec##*:}
    # One worker per task id, as in the Kubernetes job: they share this GPU's vLLM server.
    span=$(( (hi - lo + (hi - lo) - 1) / (hi - lo) ))   # = 1 task per worker
    for ((w = 0; w < hi - lo; w++)); do
      wlo=$((lo + w * span)); whi=$((wlo + span)); [ "$whi" -gt "$hi" ] && whi=$hi
      ( cd "$step_dir" || exit 1
        VLLM_API_BASE="http://127.0.0.1:$(vllm_port "$gpu")/v1" \
        python3 "$REPO/mcts_collection.py" \
          --env_server_base "http://127.0.0.1:$(env_port "$gpu")" \
          --model_name "$MODEL_NAME" --min "$wlo" --max "$whi" $extra \
          >> "$LOG_DIR/search.i$it.$task.g$gpu.w$w.log" 2>&1 ) &
      pids+=($!)
    done
    gpu=$((gpu + 1))
  done
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  stop_backends

  local count
  count=$(find "$trees" -name 'search_results_*.json' | wc -l | tr -d ' ')
  [ "$rc" -eq 0 ] || die "search $task: a worker exited non-zero; see $LOG_DIR/search.i$it.$task.*.log"
  [ "$count" -ge "$expected" ] || die "search $task: $count trees, expected $expected"
  mark_done "$step_dir" "{\"trees\": $count, \"model_dir\": \"$model\"}"
  log "iter$it search $task: $count trees"
}

# ---------------------------------------------------------------- 2. revise
step_revise() {
  local it=$1 task=$2 model=$3
  local step_dir=$RUN_DIR/iter$it/revise-$task
  done_file "$step_dir" && { log "iter$it revise $task: done, skipping"; return 0; }
  local trees=$RUN_DIR/iter$it/search-$task/mcts_result/$task/$MODEL_NAME
  local alpha; alpha=$(nth "$it" $ALPHAS)
  mkdir -p "$step_dir"

  # One directory per tree, so one path_collection.py process handles one tree, as the Job does.
  local files=() f
  while IFS= read -r f; do files+=("$f"); done < <(find "$trees" -name 'search_results_*.json' | sort)
  [ ${#files[@]} -gt 0 ] || die "revise $task: no trees under $trees"
  log "iter$it revise $task: ${#files[@]} trees over $GPUS shards, alpha=$alpha beta=$BETA pair_shards=$PAIR_SHARDS"

  export_inference_env "$task" "$model" Raw "$TEMP"
  export ALPHA=$alpha BETA=$BETA
  start_backends "$task" "$model" no || die "revise $task: vLLM did not start"

  local i=0 gpu shard_in
  for ((gpu = 0; gpu < GPUS; gpu++)); do mkdir -p "$step_dir/shard$gpu/input"; done
  for f in "${files[@]}"; do
    gpu=$((i % GPUS)); shard_in=$step_dir/shard$gpu/input/$(basename "$f" .json)
    mkdir -p "$shard_in" && ln -sfn "$f" "$shard_in/$(basename "$f")"
    i=$((i + 1))
  done

  local pids=()
  for ((gpu = 0; gpu < GPUS; gpu++)); do
    ( cd "$step_dir/shard$gpu" || exit 1
      mkdir -p out done partial && ln -sfn "$REPO/mcts_utils" mcts_utils
      # Biggest tree first: the longest unit dominates the shard's wall clock.
      ls -SL input/*/*.json | xargs -n1 dirname | xargs -n1 basename | while read -r t; do
        for ((s = 0; s < PAIR_SHARDS; s++)); do echo "$t $s"; done
      done | VLLM_API_BASE="http://127.0.0.1:$(vllm_port "$gpu")/v1" \
             REPO="$REPO" LOG_DIR="$LOG_DIR" TAG="i$it.$task.g$gpu" PAIR_SHARDS="$PAIR_SHARDS" \
             xargs -P "$CONCURRENCY" -n 2 bash -c '
          t="$1"; s="$2"
          [ -d "out/$t" ] && exit 0
          [ -d "done/$t/p$s" ] && exit 0
          rm -rf "partial/$t/p$s"
          python3 "$REPO/path_collection.py" --input_dir "input/$t" --output_dir "partial/$t/p$s" \
            --data_type centric --revise 1 --pair_shard "$s" --pair_shards "$PAIR_SHARDS" \
            >> "$LOG_DIR/revise.$TAG.$t.p$s.log" 2>&1 \
            && mkdir -p "partial/$t/p$s" "done/$t" && mv "partial/$t/p$s" "done/$t/p$s"' _
      rc=$?
      # Assemble trees whose shards all finished.
      for t in $(ls input); do
        [ -d "out/$t" ] && continue
        [ "$(ls -d done/"$t"/p* 2>/dev/null | wc -l)" -eq "$PAIR_SHARDS" ] || continue
        mkdir -p "out/$t" && cat done/"$t"/p*/*_centric.jsonl > "out/$t/${TASK}_centric.jsonl" 2>/dev/null
      done
      exit $rc ) &
    pids+=($!)
  done
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  stop_backends
  [ "$rc" -eq 0 ] || die "revise $task: see $LOG_DIR/revise.i$it.$task.*.log"

  local rows
  rows=$(cat "$step_dir"/shard*/out/*/*_centric.jsonl 2>/dev/null | wc -l | tr -d ' ')
  mark_done "$step_dir" "{\"rows\": $rows, \"alpha\": $alpha, \"beta\": $BETA}"
  log "iter$it revise $task: $rows rows"
}

# ---------------------------------------------------------------- 3. sft-data
# The controller's own implementation, so the caps, the ICL stripping, the ShareGPT mixing and the
# seeded sampling are identical rather than a second copy that can drift.
step_sft_data() {
  local it=$1
  local step_dir=$RUN_DIR/iter$it/sft-data
  done_file "$step_dir" && { log "iter$it sft-data: done, skipping"; echo "$step_dir/train.jsonl"; return 0; }
  python3 "$REPO/pipeline/controller.py" --config "$CONFIG" --code-dir "$REPO" \
    --sha "$(git -C "$REPO" rev-parse --short=12 HEAD 2>/dev/null || echo local)" \
    --step sft-data --iteration "$it" >/dev/null || die "sft-data failed"
  log "iter$it sft-data: $(cat "$step_dir/stats.json" | tr -d '\n ')"
  echo "$step_dir/train.jsonl"
}

# ---------------------------------------------------------------- 4. sft
step_sft() {
  local it=$1 model=$2 data=$3
  local step_dir=$RUN_DIR/iter$it/sft out=$RUN_DIR/iter$it/sft/output
  if done_file "$step_dir"; then
    log "iter$it sft: done, skipping"
    latest_checkpoint "$out"
    return 0
  fi
  local epochs; epochs=$(nth "$it" $SFT_EPOCHS)
  log "iter$it sft: from $model, $epochs epoch(s), batch $SFT_PER_DEVICE_BATCH x $SFT_GRAD_ACCUM x $GPUS GPUs"
  rm -rf "$out" && mkdir -p "$out"

  NPROC_PER_NODE=$GPUS \
  NCCL_NVLS_ENABLE=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HOME=$AGENTR_HF_HOME HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  "$AGENTR_SWIFT" sft \
    --model "$model" \
    --tuner_type full \
    --dataset "$data" \
    --torch_dtype "$SFT_DTYPE" \
    --max_length "$SFT_MAX_LENGTH" \
    --truncation_strategy "$SFT_TRUNCATION_STRATEGY" \
    --packing "$SFT_PACKING" \
    --loss_scale "$SFT_LOSS_SCALE" \
    --num_train_epochs "$epochs" \
    --per_device_train_batch_size "$SFT_PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$SFT_GRAD_ACCUM" \
    --learning_rate "$SFT_LEARNING_RATE" \
    --lr_scheduler_type cosine \
    --warmup_ratio "$SFT_WARMUP_RATIO" \
    --weight_decay "$SFT_WEIGHT_DECAY" \
    --max_grad_norm "$SFT_MAX_GRAD_NORM" \
    --adam_beta1 "$SFT_ADAM_BETA1" \
    --adam_beta2 "$SFT_ADAM_BETA2" \
    --deepspeed "$SFT_DEEPSPEED" \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --split_dataset_ratio 0 \
    --save_strategy epoch \
    --save_only_model true \
    --save_total_limit 1 \
    --logging_steps 5 \
    --dataset_num_proc 8 \
    --seed "$SEED" \
    --report_to tensorboard \
    --add_version false \
    --output_dir "$out" 2>&1 | tee "$LOG_DIR/sft.i$it.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "sft failed, see $LOG_DIR/sft.i$it.log"

  local ckpt; ckpt=$(latest_checkpoint "$out")
  [ -n "$ckpt" ] && [ -f "$ckpt/config.json" ] || die "sft: no usable checkpoint under $out"
  mark_done "$step_dir" "{\"checkpoint\": \"$ckpt\", \"base\": \"$model\"}"
  # ms-swift (transformers 5.x) writes config/tokenizer files the vLLM runtime may not parse.
  # Full fine-tuning changes weights, never shapes, so the base model's files describe the
  # checkpoint exactly. Same operation the Kubernetes controller performs; idempotent.
  python3 "$REPO/pipeline/controller.py" --config "$CONFIG" --code-dir "$REPO" --sha local \
    --step prepare-ckpt --checkpoint "$ckpt" >&2 || die "could not prepare $ckpt for serving"
  echo "$ckpt"
}

# ---------------------------------------------------------------- 5. eval
step_eval() {
  local it=$1 model=$2
  local suffix=${EVAL_TAG:+-$EVAL_TAG}
  local step_dir=$RUN_DIR/iter$it/eval$suffix
  done_file "$step_dir" && { log "iter$it eval: done, $(cat "$step_dir/.done")"; return 0; }
  local model_type="agentr-iter$it${EVAL_TAG:+-$EVAL_TAG}"
  local task
  for task in $TASKS; do
    log "iter$it eval $task: $model (max_steps=$EVAL_MAX_STEPS, temp=$EVAL_TEMP, tag='${EVAL_TAG:-none}')"
    mkdir -p "$step_dir/$task"
    export SCIWORLD_SPLIT=$EVAL_SCIWORLD_SPLIT
    # fixed = the paper's flat round limit (--max_steps); task = Co-Evolving's per-task budget.
    export STEP_BUDGET_MODE=$EVAL_STEP_BUDGET_MODE
    export_inference_env "$task" "$model" "$model_type" "$EVAL_TEMP"
    [ "$EVAL_TASK_LIMIT" -gt 0 ] && export TASK_LIMIT=$EVAL_TASK_LIMIT
    start_backends "$task" "$model" || die "eval $task: backends did not start"
    ( cd "$step_dir/$task" && ln -sfn "$REPO/mcts_utils" mcts_utils )
    local pids=() gpu
    # The test ids are striped over the GPUs; all shards write to one result directory and skip
    # items that already have a file, so they never collide.
    for ((gpu = 0; gpu < GPUS; gpu++)); do
      ( cd "$step_dir/$task" || exit 1
        TASK_SHARD=$gpu TASK_SHARDS=$GPUS \
        VLLM_API_BASE="http://127.0.0.1:$(vllm_port "$gpu")/v1" \
        python3 "$REPO/eval.py" --env_server_base "http://127.0.0.1:$(env_port "$gpu")" \
          --model_name "$MODEL_NAME" --max_steps "$EVAL_MAX_STEPS" \
          >> "$LOG_DIR/eval.i$it.$task.g$gpu.log" 2>&1 ) &
      pids+=($!)
    done
    local rc=0 p
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    stop_backends
    [ "$rc" -eq 0 ] || die "eval $task: see $LOG_DIR/eval.i$it.$task.*.log"
  done
  local summary
  summary=$(python3 "$REPO/pipeline/controller.py" --config "$CONFIG" --code-dir "$REPO" --sha local \
              --step score --iteration "$it") || die "scoring failed"
  mark_done "$step_dir" "$summary"
  log "iter$it eval: $(echo "$summary" | tr -d '\n ')"
}

# ---------------------------------------------------------------- driver
log "run=$RUN config=$CONFIG gpus=$GPUS tasks=$TASKS iterations=$ITERATIONS"
log "outputs: $RUN_DIR (per-step logs in $LOG_DIR)"
MODEL_DIR=$BASE_MODEL_DIR
for ((IT = 1; IT <= ITERATIONS; IT++)); do
  log "===== iteration $IT: model $MODEL_DIR"
  for TASK_NAME in $TASKS; do step_search "$IT" "$TASK_NAME" "$MODEL_DIR"; done
  for TASK_NAME in $TASKS; do step_revise "$IT" "$TASK_NAME" "$MODEL_DIR"; done
  DATA=$(step_sft_data "$IT") || exit 1
  START=$MODEL_DIR; [ "$SFT_CONTINUE" = 0 ] && START=$BASE_MODEL_DIR
  MODEL_DIR=$(step_sft "$IT" "$START" "$DATA") || exit 1
  step_eval "$IT" "$MODEL_DIR"
done
log "PIPELINE_DONE"
