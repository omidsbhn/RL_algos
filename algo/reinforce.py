from torch.autograd import Variable
from model import Policy, Value
import torch
import torch.optim as optim


# REINFORCE: Mont Carlo Policy Gradient
class REINFORCE:
    def __init__(self, in_features, n_actions, alpha=3e-4, gamma=0.99,):
        # super().__init__(in_features, n_actions)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.policy = Policy(in_features, n_actions, alpha).to(self.device)
        # print(self.device)

    def get_action(self, state):
        state = Variable(torch.Tensor(state))
        state = torch.unsqueeze(state, 0)
        probs = self.policy.forward(state.to(self.device))
        probs = torch.squeeze(probs, 0)
        action = probs.multinomial(2)
        action = action.data
        action = action[0]
        return action

    def pi(self, s, a):
        s = Variable(torch.Tensor([s]))
        probs = self.policy.forward(s.to(self.device))
        probs = torch.squeeze(probs, 0)
        return probs[a].to(self.device)

    def update_weight(self, states, actions, rewards):
        G = Variable(torch.Tensor([0])).to(self.device)
        # for each step of the episode t = T - 1, ..., 0
        # r_tt represents r_{t+1}
        for s_t, a_t, r_tt in zip(states[::-1], actions[::-1], rewards[::-1]):
            G = torch.Tensor([r_tt]).to(self.device) + self.gamma * G
            loss = (-1.0) * self.gamma * G * torch.log(self.pi(s_t, a_t))

            # update policy parameter \theta
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()


# REINFORCE with Baseline (episodic)
class ReinforceWithBaseline(REINFORCE):
    def __init__(self, in_features, n_actions, gamma=0.99, alpha_1=3e-4, alpha_2=3e-4):
        super().__init__(in_features, n_actions, alpha=alpha_1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.value = Value(in_features, alpha_2).to(self.device)
        self.gamma = gamma

    def get_value(self, state):
        state = Variable(torch.Tensor(state))
        state = torch.unsqueeze(state, 0)
        value = self.value.forward(state.to(self.device))
        return value

    def update_weight(self, states, actions, rewards):
        G = Variable(torch.Tensor([0])).to(self.device)
        # for each step of the episode t = T - 1, ..., 0
        # r_tt represents r_{t+1}
        for s_t, a_t, r_tt in zip(states[::-1], actions[::-1], rewards[::-1]):
            G = Variable(torch.Tensor([r_tt])).to(self.device) + self.gamma * G
            delta = G - self.get_value(s_t)
            value_loss = (-1.0) * delta * self.get_value(s_t)
            # print("value loss",value_loss)
            policy_loss = (-1.0) * self.gamma * delta.data[0] * torch.log(self.pi(s_t, a_t))
            # print("policy loss", policy_loss)

            # update policy and value parameter
            self.value.optimizer.zero_grad()
            value_loss.backward()
            self.value.optimizer.step()
            #
            self.policy.optimizer.zero_grad()
            policy_loss.backward()
            self.policy.optimizer.step()
