"""InterCode-SQL environment server on the ETO / Co-Evolving protocol.

Why this exists: Agent-R talks to an HTTP environment, and InterCode's SqlEnv expects to manage its
own Docker container. Inside Kubernetes we cannot run Docker, but we do not need to: the container
is only ever a MySQL server. `IntercodeEnv.__init__` touches Docker in exactly one line
(`get_container`), `reset_container()` is a no-op for SQL, and `exec_action` / `get_reward` are pure
`mysql.connector` calls. So we stub that one line and point their unmodified code at a MySQL
sidecar loaded from the same spider_all.sql their Dockerfile uses.

Nothing about scoring is reimplemented: reward is InterCode's own `get_reward`, the
intersection-over-union between the agent's result rows and the gold query's. It is continuous in
[0, 1], which is also what makes the paper's alpha (0.5 / 0.7 / 1.0) meaningful as written.

Protocol details taken from Co-Evolving (eval_agent/envs/intercode_sql_env.py and
configs/task/intercode_sql.json), so a number here is comparable with their table:
  actions      exactly one "Action:", holding either a fenced ```sql block with ONE read-only
               statement, or exactly "submit". Parsing is their parse_sql_action, imported.
  observation  the result rows, truncated to 350 characters; "" becomes
               "[Executed Successfully with No Output]".
  reward       0 until the agent submits; the IoU of the submitted result set at submit.
               Hitting the step limit without submitting scores 0.
  read-only    each episode runs in a read-only transaction, so a query that slipped past the
               parser still cannot mutate the database.
  tasks        positions in their train_indices.json (1,500) / test_indices.json (200), which index
               ic_spider_train.json (4,555 records) / ic_spider_test.json (1,015).

Endpoints are the four Agent-R needs: /create /reset /step /observation.
"""
import json
import os
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ETO_ROOT = os.environ.get("INTERCODE_ETO_ROOT", "/data/src/intercode-sql-eto")
DATA = f"{ETO_ROOT}/eval_agent/data/intercode_sql"
SPLIT = os.environ.get("INTERCODE_SPLIT", "train")
OBS_LIMIT = int(os.environ.get("INTERCODE_OBS_LIMIT", "350"))

# Point InterCode at the MySQL sidecar before importing anything that reads SQL_CONFIG.
import intercode.envs.sql.sql_env as _sql_env  # noqa: E402
import intercode.envs.ic_env as _ic_env  # noqa: E402

_sql_env.SQL_CONFIG.update({
    "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "admin"),
    "password": os.environ.get("MYSQL_PASSWORD", "admin"),
})


class _NoContainer:
    """Stands in for the Docker container object. InterCode only calls .stop() on it, and for SQL
    reset_container() is a no-op - the database is the sidecar and outlives every episode."""

    def stop(self):
        pass


_ic_env.get_container = lambda *a, **k: _NoContainer()   # the one Docker call in the whole path

from intercode.envs import SqlEnv  # noqa: E402
from intercode.utils import data_loader as _data_loader  # noqa: E402

# Every SqlEnv builds its own IntercodeDataLoader, which reads the 6 MB spider file through pandas.
# With one session per worker that is ~43 identical copies per pod, and 43 slow startups. The data
# is read-only, so cache the parsed frame per path and hand every session the same one.
_FRAMES = {}
_load_data_once = _data_loader.IntercodeDataLoader._load_data


def _cached_load_data(self):
    if self.data_path not in _FRAMES:
        _FRAMES[self.data_path] = _load_data_once(self)
    return _FRAMES[self.data_path]


_data_loader.IntercodeDataLoader._load_data = _cached_load_data

from eval_agent.intercode_sql_action import parse_sql_action  # noqa: E402  (Co-Evolving's parser)

app = FastAPI()


def preprocess_sql(record: Dict) -> str:
    """Co-Evolving's preprocess, verbatim: select the task's database before the episode."""
    return f"use {record['extra']['db']}"


def load_split(split):
    name = "train" if split == "train" else "test"
    return json.load(open(f"{DATA}/{name}_indices.json"))


def stable_unique(items):
    """Co-Evolving's stable_unique: their index lists contain duplicates, and dropping them changes
    which task a position refers to. Order-preserving, as theirs is."""
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


class Sessions:
    """One SqlEnv per session: each holds its own MySQL connection and read-only transaction, so
    concurrent Agent-R workers cannot see each other's statements."""

    def __init__(self):
        self.envs, self.info, self.next_id = {}, {}, 0
        name = "train" if SPLIT == "train" else "test"
        self.data_path = f"{DATA}/ic_spider_{name}.json"
        self.tasks = stable_unique(load_split(SPLIT))
        print(f"intercode-sql: split={SPLIT} tasks={len(self.tasks)} data={self.data_path} "
              f"mysql={_sql_env.SQL_CONFIG['host']}:{_sql_env.SQL_CONFIG['port']}", flush=True)

    def create(self):
        idx = self.next_id
        self.next_id += 1
        self.envs[idx] = SqlEnv("docker-env-sql", data_path=self.data_path,
                                preprocess=preprocess_sql, verbose=False)
        self.info[idx] = {"observation": "", "reward": 0.0, "score": 0.0, "done": False}
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


def _reconnect(env):
    """Replace this session's MySQL connection. InterCode stores the error text on the env and
    raises a generic RuntimeError, so a broken connection is otherwise indistinguishable from a
    bad query - and a session that loses its connection can never reset again."""
    import mysql.connector
    for handle in ("cur", "cnx"):
        try:
            getattr(env, handle).close()
        except Exception:
            pass
    env.cnx = mysql.connector.connect(**_sql_env.SQL_CONFIG)
    env.cur = env.cnx.cursor(buffered=True)


def _begin_episode(env, record_idx):
    # End the previous episode and make the new one read-only, as Co-Evolving's env.reset does:
    # even if an unsafe statement bypasses the parser, MySQL rejects persistent mutation.
    try:
        env.cnx.rollback()
    except Exception:
        pass
    env.reset(record_idx)                    # runs "use <db>"; raises if that fails
    env.cnx.start_transaction(readonly=True)


@app.post("/reset")
def reset(body: ResetBody):
    env = S.envs[body.id]
    record_idx = S.tasks[int(body.data_idx) % len(S.tasks)]
    try:
        _begin_episode(env, record_idx)
    except Exception as first:
        # Under load (43 workers share one server here) a connection can be dropped or left with a
        # transaction MySQL will not let us leave. Rebuild the session and try once more, and if
        # that fails too, report the MySQL text InterCode hid on env.observation instead of its
        # generic "Preprocess command failed".
        detail = getattr(env, "observation", None)
        print(f"reset({record_idx}) failed: {first!r} mysql={detail!r}; reconnecting", flush=True)
        _reconnect(env)
        try:
            _begin_episode(env, record_idx)
        except Exception as second:
            raise HTTPException(
                status_code=503,
                detail=(f"reset({record_idx}) failed twice: {second}; "
                        f"mysql said: {getattr(env, 'observation', None)!r}")) from second
    S.info[body.id] = {
        "observation": env.query,          # the natural-language question
        "record_idx": record_idx,
        "reward": 0.0, "score": 0.0, "done": False,
    }
    return S.info[body.id]


@app.post("/step")
def step(body: StepBody):
    env, st = S.envs[body.id], S.info[body.id]
    try:
        action, is_submit = parse_sql_action(body.action)
    except Exception:
        # Co-Evolving's wording, verbatim: the agent is told exactly what shape is expected.
        st["observation"] = ('I don\'t understand your input.\n Your input should include exactly '
                             'one "Action:".\n To run a read-only query, use:\nAction:\n```sql\n'
                             'SELECT ...\n```\nTo submit, use exactly: Action: submit.')
        return st

    output, reward, done, _ = env.step(action)
    if is_submit:
        st.update({"observation": str(output), "reward": float(reward),
                   "score": float(reward), "done": True})
        return st

    if isinstance(output, str) and "Error" in output and "Unknown" not in output:
        observation = f"{output}\n"
    elif output == "" or output is None:
        observation = "[Executed Successfully with No Output]"
    else:
        observation = f"{output}"
    if len(observation) > OBS_LIMIT:
        observation = observation[:OBS_LIMIT] + "..."
    st.update({"observation": observation, "reward": float(reward),
               "score": float(reward), "done": bool(done)})
    return st


@app.get("/observation")
def observation(env_idx: int):
    return S.info[env_idx]["observation"]


@app.get("/task_count")
def task_count():
    return {"count": len(S.tasks)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "36001")))
