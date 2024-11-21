import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())

        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DeepONet(nn.Module):
    def __init__(self, trunk_input_size, branch_input_size, hidden_size, output_size_branch, output_size_trunk):
        super(DeepONet, self).__init__()
        self.trunk_net = MLP(trunk_input_size, hidden_size, output_size_trunk)
        self.branch_net = MLP(branch_input_size, hidden_size, output_size_branch)

    def forward(self, x_trunk, x_branch, domain_sizes):
        output_trunk = self.trunk_net(x_trunk)

        output_branch = self.branch_net(x_branch)

        output1 = torch.sum(torch.mul(output_trunk, output_branch[:, :50]), dim=1, keepdim=True) # this is u_x
        output2 = torch.sum(torch.mul(output_trunk, output_branch[:, 50:]), dim=1, keepdim=True) # this is u_y

        # We modify the outputs with the sine functions
        output1 = output1 * self.sine(x_trunk[:,1], domain_sizes[:,0]) * self.cosine(x_trunk[:,2], domain_sizes[:,1])
        output2 = output2 * self.sine(x_trunk[:,2], domain_sizes[:,1]) * self.cosine(x_trunk[:,1], domain_sizes[:,0])

        return torch.cat((output1, output2), dim=1)

    def sine(self, x, size_dom):
      return torch.sin((np.pi*x.unsqueeze(1))/size_dom.unsqueeze(1))

    def cosine(self, x, size_dom):
      return torch.cos((np.pi*x.unsqueeze(1))/(2*size_dom.unsqueeze(1)))