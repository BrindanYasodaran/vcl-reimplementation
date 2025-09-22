from .vcl import BayesianLinear
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

        return F.log_softmax(logits, dim=-1), kl_loss

    def predict(self, x, num_samples=10):
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            # Collect log-probabilities from each sample pass
            # We only need the model output (index 0), not the KL loss
            log_probs_samples = [self.forward(x)[0] for _ in range(num_samples)]
            
            # Stack the log-probabilities into a single tensor
            log_probs_tensor = torch.stack(log_probs_samples, dim=0)
            
            # Convert to probabilities, average, and convert back to log-probabilities
            avg_log_probs = torch.log(torch.exp(log_probs_tensor).mean(dim=0))

        return avg_log_probs