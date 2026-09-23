"""MCTS over InterCode-SQL, following mcts_utils/webshop/mcts_ws.py.

The search is structurally the WebShop one: the environment takes a text action and returns
(observation, reward, done), and because the environment cannot be copied, every generation replays
the episode from reset() before stepping. Three things differ, all forced by the task:

  actions       the reply is passed to the environment UNSPLIT. An InterCode action is a fenced
                ```sql block, so splitting on "Action:" as the WebShop path does would cut the
                fence off and the parser would reject every query.
  reward        InterCode scores only at submit (intersection over union of the result rows), so a
                node's env_score is 0 until the agent submits, like WebShop's terminal reward at
                "Buy Now" - not a per-step signal.
  observations  already truncated to 350 characters by the server, matching Co-Evolving.
"""
from copy import deepcopy
from dataclasses import dataclass
import mmengine
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from mcts_utils.mcts_raw import MCTSNode, MCTSAgent
import os
import time as _time
from collections import defaultdict as _dd

_PROF = os.environ.get("MCTS_PROFILE", "0").lower() in ("1", "true", "yes")
_T, _N = _dd(float), _dd(int)


def _tick():
    return _time.perf_counter() if _PROF else 0.0


def _tock(key, t0):
    if _PROF:
        _T[key] += _time.perf_counter() - t0
        _N[key] += 1


def _prof_report(tag):
    if not _PROF:
        return
    tot = sum(_T.values()) or 1.0
    parts = " ".join(f"{k}={_T[k]:.1f}s/{_N[k]}({100*_T[k]/tot:.0f}%)" for k in sorted(_T, key=lambda x: -_T[x]))
    print(f"MCTS_PROFILE {tag} total_measured={tot:.1f}s {parts}", flush=True)


@dataclass
class MCTSConfig:
    max_depth: int = int(os.environ["MAX_DEPTH"])
    iterations: int = int(os.environ["ITERA"])
    n_generate_samples: int = int(os.environ["N_GEN"])
    coef = 0.25


class ExtendedNode(MCTSNode):
    def __init__(self,
                 env=None,
                 recent_actions=None,
                 action="",
                 obs="",
                 disaster=False,
                 env_score=0,
                 puct_value=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.env_score = env_score
        self.recent_actions = recent_actions
        self.action = action
        self.disaster = disaster
        self.puct_value = puct_value
        self.obs = obs

    @property
    def reward(self):
        return self.env_score

    def to_dict(self):
        return {
            'visits': self.visits,
            'value': self.value,
            'prior': self.prior,
            'puct_value': self.puct,
            'obs': self.obs,
            'llm_response': self.llm_response,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'recent_actions': self.recent_actions,
            'action': self.action,
            'env_score': self.env_score,
            'disaster': self.disaster,
            'state': self.state.to_openai_api_messages(),
            'children': [child.to_dict() for child in self.children]
        }


class ExtendedMCTS(MCTSAgent):
    def __init__(self,
                 idx=0,
                 calling=None,
                 encoding=None,
                 max_len=0,
                 model_name=None,
                 logger=None,
                 env=None,
                 generate_cfg=MCTSConfig()):
        super().__init__()
        self.generate_cfg = generate_cfg
        self.calling = calling
        self.encoding = encoding
        self.max_len = max_len
        self.model_name = model_name
        self.logger = logger
        self.env = env
        self.idx = idx

    def search(self, env, conv, recent_actions):
        recent_actions_temp = []
        env_reward = 0
        env_done = False
        for agent_response in recent_actions:
            conv.append_message(conv.roles[1], None)
            step_output = self.env.step(agent_response)      # unsplit: the action is a SQL fence
            env_reward, env_done = step_output.reward, step_output.done
            current_obs = step_output.state
            recent_actions_temp.append([agent_response, current_obs])
            conv.update_last_message(agent_response)
            conv.append_message(conv.roles[0], current_obs)

        init_state = deepcopy(conv)
        self.root = ExtendedNode(env=env, state=init_state, llm_response="ROOT",
                                 is_terminal=env_done, recent_actions=recent_actions_temp,
                                 env_score=env_reward, action="ROOT")

        for iter in range(self.generate_cfg.iterations):
            node = self.root
            if node.is_terminal:
                print(f"Stop at Iter {iter}")
                return
            while node and not node.is_terminal:
                self.expand(node)
                node = self._select(node)
        _prof_report(f"task={self.idx}")
        return

    def _prompt(self, node):
        # As in the released code: drop the oldest exchange until the prompt fits. Index 4 is the
        # first turn after the instruction, "OK" and the in-context example's opening, so the
        # instruction and the example's framing survive truncation.
        _t = _tick()
        conv = deepcopy(node.state)
        while len(self.calling.encoding.encode(str(conv))) > self.max_len - 60:
            if len(conv.messages) <= 6:
                break                                  # nothing left to drop but the framing
            del conv.messages[4:6]
        out = conv.to_openai_api_messages()
        _tock("prompt_build", _t)
        return out

    def _generate(self, node, agent_response=None):
        if agent_response is None:
            prompt = self._prompt(node)
            _t = _tick()
            agent_response = self.calling.llm_func(prompt, self.model_name)
            _tock("llm", _t)
        disaster = False
        agent_response = agent_response.strip()
        _t = _tick()
        conv = deepcopy(node.state)
        _tock("deepcopy", _t)

        # The environment holds one MySQL session, so it cannot be forked: replay this node's
        # actions from reset() to put the database and the episode back in the right state.
        _t = _tick()
        _ = self.env.reset(self.idx)
        conv.append_message(conv.roles[1], None)
        conv.update_last_message(agent_response)
        for action in node.recent_actions:
            _ = self.env.step(action[0])
        _tock("env_replay", _t)

        current_env = self.env
        current_recent_actions = deepcopy(node.recent_actions)

        _t = _tick()
        step_output = current_env.step(agent_response)   # unsplit; the server parses the fence
        _tock("env_step", _t)
        current_obs, new_env_score, done = step_output.state, step_output.reward, step_output.done

        if new_env_score < 0:
            is_terminal = True
            new_env_score = 0
            disaster = True
        else:
            is_terminal = done

        current_recent_actions.append([agent_response, current_obs])

        print(agent_response)
        print(current_obs)
        print(new_env_score)
        conv.append_message(conv.roles[0], current_obs)
        new_node = ExtendedNode(
            obs=current_obs,
            action=agent_response,
            env=current_env,
            state=conv,
            parent=node,
            disaster=disaster,
            recent_actions=current_recent_actions,
            llm_response=agent_response,
            depth=node.depth + 1,
            env_score=new_env_score,
            is_terminal=node.depth + 1 > self.generate_cfg.max_depth or is_terminal
        )
        return new_node

    def expand(self, node):
        if not node.is_fully_expanded:
            n = self.generate_cfg.n_generate_samples
            if os.environ.get("MCTS_BATCH_GEN", "0").lower() in ("1", "true", "yes"):
                _p = self._prompt(node)
                _t = _tick()
                replies = self.calling.llm_func_n(_p, self.model_name, n)
                _tock("llm", _t)
                sampled_nodes = [self._generate(node, agent_response=r) for r in replies]
            else:
                sampled_nodes = [self._generate(node) for _ in range(n)]
            fingerprint, dedup_nodes = set(), []

            for sample_node in sampled_nodes:
                if sample_node.llm_response in fingerprint:
                    continue
                fingerprint.add(sample_node.llm_response)
                dedup_nodes.append(sample_node)
            node.children = dedup_nodes
            for child in node.children:
                if child.is_terminal:
                    self._backpropagate(child, child.reward)

    def clean(self, s):
        # WebShop flattens newlines; SQL observations are tabular and the newlines carry meaning,
        # and the server has already capped the length, so leave them alone.
        return s

    def load(data_path):
        state_dict = mmengine.load(data_path)

        def dict_to_node(data):
            children_data = data.pop('children')
            node = ExtendedNode(**data)
            node.children = [dict_to_node(child) for child in children_data]
            return node
