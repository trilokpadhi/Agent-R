"""ScienceWorld environment server on the ETO / Co-Evolving protocol.

Why this exists: Agent-R talks to an HTTP environment, Co-Evolving runs ScienceWorld in-process. The
table we must match was produced by the latter, and the two differ in ways that change the score:

  simplification   Co-Evolving loads every task with simplificationStr="easy" (teleport between
                   rooms, containers pre-opened). AgentGym's copy loads with no simplification.
  reward scale     Co-Evolving reads info["raw_score"], a 0-1 float, and keeps the MAXIMUM over the
                   episode. AgentGym returns info["score"], a 0-100 int, read at the end. This also
                   makes the paper's alpha (0.5/0.7/1.0) meaningful as written - on a 0-100 scale
                   every path clears it and the filter does nothing.
  task addressing  Co-Evolving uses (task_name, variation_idx) pairs from its own split files;
                   AgentGym uses a flat index into all 4,639 (task, variation) pairs.
  step budget      per task, from max_steps.json (10-120, mean 39.7 over the test set), not one
                   global number.

Scoring is not reimplemented here: this imports Co-Evolving's own monkey patch, so raw_score,
termination_cause and the rest are computed by their code.

Runs in /data/envs/agentenv-sciworld (Python 3.8, scienceworld, Java) with
PYTHONPATH=/data/src/sciworld-eto. Nothing under /data/src/AgentGym-agentr is touched.

Endpoints: the four Agent-R always needs (create/reset/step/observation) plus the three its
SciWorld code additionally calls - look_around, inventory, valid_action_object_combinations - which
findValidActionNew uses to snap a generated action onto the legal action set.
"""
import json
import os

from fastapi import FastAPI
from pydantic import BaseModel

from eval_agent.utils.replace_sciworld_score import sciworld_monkey_patch
from scienceworld import ScienceWorldEnv

sciworld_monkey_patch()   # their step(): raw_score 0-1, termination_cause, look/inv/valid in infos

ETO_ROOT = os.environ.get("SCIWORLD_ETO_ROOT", "/data/src/sciworld-eto")
DATA = f"{ETO_ROOT}/eval_agent/data/sciworld"
SPLIT = os.environ.get("SCIWORLD_SPLIT", "train")      # search uses train, eval uses test
JAR = os.environ.get("SCIWORLD_JAR", "")
STEP_LIMIT = int(os.environ.get("SCIWORLD_ENV_STEP_LIMIT", "200"))

app = FastAPI()


def load_split(split):
    """(task_name, variation_idx) pairs. Co-Evolving maps every non-train split to test."""
    name = "train" if split == "train" else ("dev" if split == "dev" else "test")
    return [tuple(x) for x in json.load(open(f"{DATA}/{name}_indices.json"))]


class Sessions:
    """One ScienceWorldEnv per session. The Java backend holds per-env state, so sessions cannot
    share one instance the way WebShop's SimServer could."""

    def __init__(self):
        self.envs, self.info, self.next_id = {}, {}, 0
        self.tasks = load_split(SPLIT)
        self.max_steps = json.load(open(f"{DATA}/max_steps.json"))
        print(f"sciworld-eto: split={SPLIT} tasks={len(self.tasks)} jar={JAR or '(package default)'}", flush=True)

    def create(self):
        idx = self.next_id
        self.next_id += 1
        kw = {"envStepLimit": STEP_LIMIT}
        if JAR:
            kw["serverPath"] = JAR
        self.envs[idx] = ScienceWorldEnv("", **kw)
        self.info[idx] = {"observation": "", "reward": 0.0, "score": 0.0, "done": False}
        return idx

    def get(self, idx):
        if idx not in self.envs:
            raise KeyError(f"unknown env id {idx}")
        return idx


S = None


@app.on_event("startup")
def _startup():
    global S
    S = Sessions()


class ResetBody(BaseModel):
    id: int
    data_idx: int


class StepBody(BaseModel):
    id: int
    action: str


@app.get("/")
def root():
    return "ok"


@app.post("/create")
def create():
    return {"id": S.create()}


@app.post("/reset")
def reset(body: ResetBody):
    S.get(body.id)
    task_name, variation = S.tasks[int(body.data_idx) % len(S.tasks)]
    env = S.envs[body.id]
    # Co-Evolving's reset, verbatim: easy simplification, no gold path.
    env.load(task_name, variation, simplificationStr="easy", generateGoldPath=False)
    obs, info = env.reset()
    S.info[body.id] = {
        "observation": info["taskDesc"] + "\n" + obs,   # task description then first observation
        "task_description": info["taskDesc"],
        "task_name": task_name, "variation_idx": variation,
        "max_steps": S.max_steps.get(task_name),        # per-task budget, their protocol
        "reward": 0.0, "score": 0.0, "done": False,
    }
    return S.info[body.id]


@app.post("/step")
def step(body: StepBody):
    S.get(body.id)
    env, st = S.envs[body.id], S.info[body.id]
    action = body.action.strip()
    if "Action:" in action:                 # tolerate a raw reply; Agent-R normally pre-extracts
        action = action.split("Action:")[-1].strip()
    try:
        observation, _, done, info = env.step(action)
        reward = float(info["raw_score"])   # 0-1, their scale
    except AssertionError:                  # their handling of an action the engine rejects
        observation, done, reward = "Observation: Invalid action!", False, st["reward"]
    # max over the episode, as Co-Evolving does - ScienceWorld's score can fall
    st["reward"] = max(float(st.get("reward") or 0.0), reward)
    st["score"] = st["reward"]
    st["observation"] = observation
    st["done"] = bool(done)
    return st


@app.get("/observation")
def observation(env_idx: int):
    return S.info[env_idx]["observation"]


@app.get("/look_around")
def look_around(env_idx: int):
    return {"look_around": S.envs[env_idx].look()}


@app.get("/inventory")
def inventory(env_idx: int):
    return {"inventory": S.envs[env_idx].inventory()}


@app.get("/valid_action_object_combinations")
def valid_actions(env_idx: int):
    return {"get_valid_action_object_combinations": S.envs[env_idx].getValidActionObjectCombinations()}


@app.get("/task_count")
def task_count():
    return {"count": len(S.tasks)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "36001")))
