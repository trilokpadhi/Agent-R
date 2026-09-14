# AgentGym SciWorld files from the Agent-R maintainers

Verbatim code posted by Agent-R collaborator `siyuyuan` in GitHub issues; Agent-R's SciWorld code
(`get_look_around`, `get_inventory`, `get_valid_action_object_combinations`, `get_game_nums`) was
written against this modified AgentGym. Extracted programmatically from the issue comments.

| File | Source comment | Posted | Lines |
|---|---|---|---|
| `agentenv-sciworld/agentenv_sciworld/environment.py` | https://github.com/ByteDance-Seed/Agent-R/issues/8#issuecomment-2961218292 | 2025-06-11T04:40:08Z | 156 |
| `agentenv-sciworld/agentenv_sciworld/server.py` | https://github.com/ByteDance-Seed/Agent-R/issues/8#issuecomment-2961218292 | 2025-06-11T04:40:08Z | 64 |
| `agentenv/agentenv/envs/sciworld.py` | https://github.com/ByteDance-Seed/Agent-R/issues/5#issuecomment-2732580900 | 2025-03-18T10:28:55Z | 150 |

Context: https://github.com/ByteDance-Seed/Agent-R/issues/5 and https://github.com/ByteDance-Seed/Agent-R/issues/8

## Deployment on ARCTIC

- Copied into `/data/src/AgentGym-agentr/` (copies of upstream `agentenv/` and `agentenv-sciworld/`
  from AgentGym `3ef9235`, with these three files replaced). `/data/src/AgentGym` stays upstream.
- One compatibility change in the copy, not part of the maintainer posts: current AgentGym's
  `agentenv/agentenv/envs/__init__.py` imports `SciWorldAdapter`, which the maintainer client does not
  define, so that line was changed to `from .sciworld import SciworldEnvClient, SciworldTask`.
- The SciWorld conda env has `agentenv_sciworld` installed editable from the upstream checkout, so Jobs
  start the server from `cd /data/src/AgentGym-agentr/agentenv-sciworld` with
  `python -c "import uvicorn; uvicorn.run('agentenv_sciworld:app', ...)"` and assert the import origin.
