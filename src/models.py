from vcl import BayesianLinear
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import torch

class BayesianMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianMLP, self).__init__()
        self.fc1 = BayesianLinear(in_features, 100)
        self.fc2 = BayesianLinear(100, 100)
        self.fc3 = BayesianLinear(100, out_features)

    def update_prior(self):
        self.fc1.update_prior()
        self.fc2.update_prior()
        self.fc3.update_prior()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        kl_loss = self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.fc3.kl_divergence()

        return logits, kl_loss

    def predict(self, x, num_samples=10):
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            logits_samples = [self.forward(x)[0] for _ in range(num_samples)]
            # Stack the logits into a single tensor
            logits_tensor = torch.stack(logits_samples, dim=0)
            # Apply softmax to convert logits to probabilities, then average
            avg_probs = F.softmax(logits_tensor, dim=-1).mean(dim=0)

        return torch.log(avg_probs)

class StandardMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(StandardMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits