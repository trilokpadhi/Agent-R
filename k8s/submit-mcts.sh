#!/usr/bin/env bash
# Single source for Agent-R MCTS collection on all 8 H100s.
#
# Generates the 8 shard manifests from one template and submits them in one apply.
# Nothing is edited per shard by hand: the only per-shard values are the job name and
# the task range, both set here.
#
#   k8s/submit-mcts.sh webshop            # generate + submit all 8 WebShop shards
#   k8s/submit-mcts.sh sciworld           # generate + submit all 8 SciWorld shards
#   k8s/submit-mcts.sh webshop --generate # write the manifests, do not submit
#
# Shards write to one shared results directory per environment and skip ids that already
# have a result file, so resubmitting the whole set only redoes tasks that were in flight.
set -euo pipefail

NS=ii400r87
LOGIN=tpadhi1@arclogin02.rs.gsu.edu
SSH_OPTS=(-o ControlMaster=no -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" -o BatchMode=yes)

ENV_NAME=${1:?usage: submit-mcts.sh webshop|sciworld [--generate]}
MODE=${2:-submit}

case "$ENV_NAME" in
  webshop)
    # 300 usable training ids (in range(1000), minus test ids) split 8 ways -> id ranges.
    RANGES=(0:58 59:113 113:176 177:231 231:294 294:340 341:405 405:471)
    TASK_ITERATION=""
    ;;
  sciworld)
    # The maintainers' 23 task_nums split 8 ways; 23 x 9 variations ~= 200 simulations.
    RANGES=(0:3 3:6 6:9 9:12 12:15 15:18 18:21 21:23)
    TASK_ITERATION=9
    ;;
  *)
    echo "unknown environment: $ENV_NAME (expected webshop or sciworld)" >&2
    exit 1
    ;;
esac

TEMPLATE="k8s/qwen35-${ENV_NAME}-mcts.yaml"
OUT_DIR="k8s/mcts-iter1-${ENV_NAME}8"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.yaml

i=0
for range in "${RANGES[@]}"; do
  i=$((i + 1))
  shard=$(printf "w%02d" "$i")
  name="qwen35-${ENV_NAME}-mcts-it1-${shard}"
  ruby -ryaml -e '
    template, name, range, task_iteration = ARGV
    min, max = range.split(":")
    doc = YAML.load_file(template)
    doc["metadata"]["name"] = name
    env = doc["spec"]["template"]["spec"]["containers"][0]["env"]
    set = ->(key, value) {
      entry = env.find { |e| e["name"] == key } or abort("template has no env var #{key}")
      entry["value"] = value
    }
    set.call("JOB_NAME", name)
    set.call("SHARD_MIN", min)
    set.call("SHARD_MAX", max)
    set.call("TASK_ITERATION", task_iteration) unless task_iteration.empty?

    # Guardrails: these are what actually broke earlier runs.
    node = doc["spec"]["template"]["spec"]["nodeSelector"]
    pinned = node && (node["kubernetes.io/hostname"] == "adonis-h100-8" || node["nvidia.com/gpu.product"].to_s.include?("H100"))
    abort("#{name}: not pinned to an H100 node") unless pinned
    abort("#{name}: no GPU requested") unless doc["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"].to_s == "1"

    File.write(ARGV[4], doc.to_yaml)
    puts "generated #{ARGV[4]}  #{name}  range=#{min}..#{max}"
  ' "$TEMPLATE" "$name" "$range" "$TASK_ITERATION" "$OUT_DIR/$name.yaml"
done

if [ "$MODE" = "--generate" ]; then
  echo "generate-only: $OUT_DIR"
  exit 0
fi

echo "== server dry run (all 8 at once)"
cat "$OUT_DIR"/*.yaml | ssh "${SSH_OPTS[@]}" "$LOGIN" \
  "export PATH=\"\$HOME/.local/bin:\$PATH\"; timeout 120 kubectl apply -n $NS --dry-run=server -f -"

echo "== submit (all 8 at once)"
cat "$OUT_DIR"/*.yaml | ssh "${SSH_OPTS[@]}" "$LOGIN" \
  "export PATH=\"\$HOME/.local/bin:\$PATH\"; timeout 120 kubectl apply -n $NS -f -"

echo "== pods"
ssh -n "${SSH_OPTS[@]}" "$LOGIN" \
  "export PATH=\"\$HOME/.local/bin:\$PATH\"; timeout 60 kubectl get pods -n $NS -l stage=mcts-iter1 -o custom-columns=NAME:.metadata.name,STATUS:.status.phase,NODE:.spec.nodeName --no-headers"
