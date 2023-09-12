from torch.autograd import Variable
from model import model
import torch
import torch.optim as optim


# REINFORCE: Mont Carlo Policy Gradient
class REINFORCE(model):
    def __init__(self, in_features, n_actions, gamma=0.99, alpha=3e-4):
        super().__init__(in_features, n_actions)
        self.gamma = gamma
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def get_action(self, state):
        print("----")
        state = Variable(torch.Tensor(state))
        state = torch.unsqueeze(state, 0)
        probs = self.forward(state)
        probs = torch.squeeze(probs, 0)
        action = probs.multinomial(2)
        action = action.data
        action = action[0]
        return action

    def pi(self, s, a):
        s = Variable(torch.Tensor([s]))
        probs = self.forward(s)
        probs = torch.squeeze(probs, 0)
        return probs[a]

    def update_weight(self, states, actions, rewards):
        G = Variable(torch.Tensor([0]))
        # for each step of the episode t = T - 1, ..., 0
        # r_tt represents r_{t+1}
        for s_t, a_t, r_tt in zip(states[::-1], actions[::-1], rewards[::-1]):
            G = Variable(torch.Tensor([r_tt])) + self.gamma * G
            loss = (-1.0) * G * torch.log(self.pi(s_t, a_t))

            # update policy parameter \theta
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
