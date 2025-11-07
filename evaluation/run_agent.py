from argparse import ArgumentParser
import pickle

import minerl
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
import gym

from evaluation.agent import MineRLAgent, ENV_KWARGS


def _reset_env(env, *, seed=None):
    """Handle Gymnasium/Gym reset return conventions."""
    reset_out = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(reset_out, tuple):
        obs, info = reset_out
    else:
        obs, info = reset_out, {}
    return obs, info


def _step_env(env, action):
    """Handle Gymnasium/Gym step return conventions."""
    step_out = env.step(action)
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = step_out
    return obs, reward, done, info

def main(model, weights):
    #env = HumanSurvival(**ENV_KWARGS).make()
    env = gym.make("MineRLBasaltFindCave-v0")
    print("---Loading model---")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    print("---Launching MineRL enviroment (be patient)---")
    obs, _ = _reset_env(env)

    while True:
        minerl_action = agent.get_action(obs)
        minerl_action["ESC"] = 0
        obs, reward, done, info = _step_env(env, minerl_action)
        env.render()
        if done:
            agent.reset()
            obs, _ = _reset_env(env)


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")

    args = parser.parse_args()

    main(args.model, args.weights)
