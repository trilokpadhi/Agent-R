"""WebShop environment server on the ETO / Co-Evolving protocol.

Why this exists: Agent-R runs AgentGym's WebShop, which is the 1,000-product subset with its own
200-item test split. The paper table we must match was produced on the full 1,181,430-product
catalogue with the ETO split; the two test sets overlap by 3 of 200 items. A number collected on
AgentGym cannot be compared with that table, so we serve the ETO environment instead.

Runs in /data/envs/agentenv-webshop (Python 3.8, pyserini, spaCy, Java) with
PYTHONPATH=/data/src/webshop-eto/envs/webshop/src. It does NOT import agentenv, and nothing under
/data/src/AgentGym is touched or shadowed (that package is `web_agent_site`; this one is
`webshop.web_agent_site`).

Protocol reproduced from Co-Evolving-Agents cc43f24, eval_agent/envs/webshop_env.py:
  * one heavyweight SimServer shared by every session (SharedWebShopFactory, async_controller/webshop.py)
  * `Action: <x>` is parsed out of the raw model reply with re.findall(r"Action: (.*)")[0] -- the FIRST
    match, as they do; a reply with no `Action:` gets an explicit format-error observation
  * an action that is not a valid clickable is a silent no-op returning the unchanged page, because
    WebAgentTextEnv.step falls through to `status = dict(reward=0, done=False)`
  * reward is the terminal dense score at click[Buy Now]; there is no max-over-episode
Step counting and the step budget stay on the Agent-R side (perform_test), as in the released code.
"""
import os
import re
import sys

from fastapi import FastAPI
from pydantic import BaseModel

from webshop.web_agent_site.envs import WebAgentTextEnv

ACTION_RE = re.compile(r"Action: (.*)")
INVALID_FORMAT = "Observation: Invalid format. The input must contains 'Action: '"

app = FastAPI()


class _Sessions:
    """One SimServer (catalogue, Lucene index, spaCy) shared by every session.

    Building a WebAgentTextEnv per worker would reload the 1.18M-product catalogue each time and,
    worse, redraw product prices: WebShop generates them with an unseeded global RNG inside
    load_products() and only calls random.seed(233) afterwards, so each construction invents
    different "under $X" goal caps. Sharing one server is what makes a task id mean one thing.
    """

    def __init__(self):
        self.envs = {}
        self.next_id = 0
        print("building shared WebShop server (full catalogue)...", flush=True)
        seed_env = WebAgentTextEnv(observation_mode="text", human_goals=True)
        self.server = seed_env.server
        self.server.user_sessions.pop(seed_env.session, None)
        print(f"products={len(self.server.all_products)} goals={len(self.server.goals)}", flush=True)

    def create(self):
        idx = self.next_id
        self.next_id += 1
        env = WebAgentTextEnv(observation_mode="text", human_goals=True, server=self.server)
        self.server.user_sessions.pop(env.session, None)
        self.envs[idx] = {"env": env, "obs": "", "reward": 0.0, "done": False}
        return idx

    def get(self, idx):
        if idx not in self.envs:
            raise KeyError(f"unknown env id {idx}")
        return self.envs[idx]


SESSIONS = None


@app.on_event("startup")
def _startup():
    global SESSIONS
    SESSIONS = _Sessions()


class CreateBody(BaseModel):
    pass


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
    return {"id": SESSIONS.create()}


@app.post("/reset")
def reset(body: ResetBody):
    state = SESSIONS.get(body.id)
    SESSIONS.server.user_sessions.pop(str(body.data_idx), None)
    # ETO addresses tasks by WebShop session id, an integer index into the 12,087 human goals.
    state["env"].reset(body.data_idx)
    state.update(obs=state["env"].observation, reward=0.0, done=False)
    return {"observation": state["obs"], "reward": 0.0, "score": 0.0, "done": False}


@app.post("/step")
def step(body: StepBody):
    state = SESSIONS.get(body.id)
    env = state["env"]
    # The RAW model reply arrives here, not a pre-extracted action: Co-Evolving decides the
    # format-error observation by whether `Action:` is present, which the caller cannot express
    # once it has already split the string.
    matches = ACTION_RE.findall(body.action.strip())
    if not matches:
        state["obs"] = INVALID_FORMAT
        return {"observation": INVALID_FORMAT, "reward": state["reward"], "score": state["reward"],
                "done": False, "invalid_format": True}
    action = matches[0].strip()
    observation, reward, done, _ = env.step(action)
    state["obs"] = f"Observation:\n{env.observation}"
    if done:
        state["reward"] = float(reward)
    state["done"] = bool(done)
    return {"observation": state["obs"], "reward": state["reward"], "score": state["reward"],
            "done": state["done"], "invalid_format": False}


@app.get("/observation")
def observation(env_idx: int):
    return SESSIONS.get(env_idx)["obs"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "36001")))
