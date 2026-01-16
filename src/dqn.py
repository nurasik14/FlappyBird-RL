import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, enable_dueling_dqn=False):
        super(DQN, self).__init__()

        self.enable_dueling_dqn = enable_dueling_dqn

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        if enable_dueling_dqn:
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, output_dim)
        else:
            self.output(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            x_val = F.relu(self.fc_value(x))
            x_adv = F.relu(self.fc_advantages(x))

            V = F.relu(self.value(x_val))
            A = F.relu(self.advantages(x_adv))

            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.output(x)

        return Q
    
if __name__ == "__main__":
    input_dim = 12
    output_dim = 2
    net = DQN(input_dim, output_dim)
    state = torch.rand(1, input_dim)
    output = net(state)
    print(output)