#!/usr/bin/env bash
# Launch or resume the Agent-R pipeline from the committed code. Run from the Mac:
#
#   pipeline/deploy.sh pipeline/config.yaml          # the real run
#   pipeline/deploy.sh pipeline/config-smoke.yaml    # tiny end-to-end check
#
# 1. refuses to run with uncommitted changes (a run must correspond to a commit)
# 2. copies `git archive HEAD` to /data/src/agentr-pipeline/<sha> on the PVC (once per commit)
# 3. applies the controller's permissions and the ONE controller Job
# Resuming: run the same command again; finished steps are skipped.
set -euo pipefail

CONFIG=${1:?usage: pipeline/deploy.sh pipeline/<config>.yaml}
LOGIN=tpadhi1@arclogin02.rs.gsu.edu
SSH=(ssh -o ControlMaster=no -o "ControlPath=$HOME/.ssh/cm-%r@%h:%p" -o BatchMode=yes "$LOGIN")
K='export PATH="$HOME/.local/bin:$PATH"; timeout 300 kubectl -n ii400r87'

cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain --untracked-files=no)" ] || [ -n "$(git ls-files --others --exclude-standard pipeline)" ]; then
  echo "Commit your changes first: a pipeline run must correspond to a git commit." >&2
  git status --short >&2
  exit 1
fi
git ls-files --error-unmatch "$CONFIG" >/dev/null

SHA=$(git rev-parse --short=12 HEAD)
CODE=/data/src/agentr-pipeline/$SHA
read -r RUN PVC IMAGE < <(ruby -ryaml -e 'c = YAML.load_file(ARGV[0]); puts [c["run"], c["pvc"], c["images"]["controller"]].join(" ")' "$CONFIG")
echo "run=$RUN commit=$SHA config=$CONFIG"

echo "== code on the PVC: $CODE"
if "${SSH[@]}" "$K exec access-pod -- test -f $CODE/pipeline/controller.py" </dev/null 2>/dev/null; then
  echo "already present"
else
  git archive --format=tar HEAD | "${SSH[@]}" "$K exec -i access-pod -- sh -c 'rm -rf $CODE.tmp && mkdir -p $CODE.tmp && tar x -C $CODE.tmp && mv $CODE.tmp $CODE'"
  echo "uploaded"
fi

echo "== permissions"
"${SSH[@]}" "$K apply -f -" < pipeline/rbac.yaml

echo "== controller Job agr-$RUN-controller"
STATE=$("${SSH[@]}" "$K get job agr-$RUN-controller -o jsonpath='{.status.succeeded}/{.status.failed}/{.status.active}'" </dev/null 2>/dev/null || true)
case "$STATE" in
  */*/1) echo "already running; follow it with: kubectl logs -n ii400r87 -f job/agr-$RUN-controller"; exit 0 ;;
  1/*)   echo "already finished successfully"; exit 0 ;;
  */1/*) echo "previous controller failed; recreating to resume"
         "${SSH[@]}" "$K delete job agr-$RUN-controller --wait=true" </dev/null ;;
esac
sed -e "s|{{RUN}}|$RUN|g" -e "s|{{SHA}}|$SHA|g" -e "s|{{CODE}}|$CODE|g" -e "s|{{CONFIG}}|$CONFIG|g" \
    -e "s|{{PVC}}|$PVC|g" -e "s|{{IMAGE_CONTROLLER}}|$IMAGE|g" pipeline/templates/controller-job.yaml \
  | "${SSH[@]}" "$K apply -f -"
echo "follow: kubectl logs -n ii400r87 -f job/agr-$RUN-controller"
echo "outputs: /data/runs/$RUN (per-step logs in /data/runs/$RUN/logs)"
