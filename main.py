from __future__ import print_function

from algo.reinforce import REINFORCE
import gym
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

MAX_EPISODES = 1500


writer = SummaryWriter()


def main():
    env = gym.make('CartPole-v1')
    agent = REINFORCE(env.observation_space.shape[0], 2)
    rew = []
    for i_episode in range(MAX_EPISODES):
        state, _ = env.reset()
        states = []
        actions = []
        rewards = [0]  # no reward at t = 0
        t = 0
        while True:
            action = agent.get_action(state)
            states.append(state)
            actions.append(action)
            state, reward, done, terminate, _ = env.step(action.numpy())
            rewards.append(reward)
            t += 1
            if done or terminate:
                writer.add_scalar("Episode/Reward", sum(rewards), i_episode)
                writer.add_scalar("Episode/Length", t, i_episode)
                break

        agent.update_weight(states, actions, rewards)

    env.close()
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
