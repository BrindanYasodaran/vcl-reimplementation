import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters for the posterior distribution (q_t)
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = Parameter(torch.Tensor(out_features))
        self.bias_rho = Parameter(torch.Tensor(out_features))

        # Non-learnable buffers for the prior distribution (q_{t-1})
        # We store mu and sigma directly, as they are fixed during training on a task
        self.register_buffer('prior_weight_mu', torch.zeros_like(self.weight_mu))
        self.register_buffer('prior_bias_mu', torch.zeros_like(self.bias_mu))
        self.register_buffer('prior_weight_sigma', torch.ones_like(self.weight_rho))
        self.register_buffer('prior_bias_sigma', torch.ones_like(self.bias_rho))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the posterior means using a standard method
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.bias_mu.data.uniform_(-stdv, stdv)
        # Initialize rho for a small initial standard deviation (e.g., 0.05)
        self.weight_rho.data.fill_(-2.9) # log(exp(0.05) - 1) approx -2.9
        self.bias_rho.data.fill_(-2.9)

    def update_prior(self):
        """
        Copies the current posterior parameters to the prior buffers.
        This should be called after training on a task is complete.
        """
        with torch.no_grad():
            self.prior_weight_mu.copy_(self.weight_mu.data)
            self.prior_bias_mu.copy_(self.bias_mu.data)
            self.prior_weight_sigma.copy_(F.softplus(self.weight_rho.data))
            self.prior_bias_sigma.copy_(F.softplus(self.bias_rho.data))

    def kl_divergence(self):
        """
        Computes the KL divergence between the posterior (self) and the prior.
        KL(q || p) = log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q-mu_p)^2)/(2*sigma_p^2) - 0.5
        """
        posterior_weight_sigma = F.softplus(self.weight_rho)
        posterior_bias_sigma = F.softplus(self.bias_rho)

        kl_weights = (torch.log(self.prior_weight_sigma / posterior_weight_sigma)
                      + (posterior_weight_sigma.pow(2) + (self.weight_mu - self.prior_weight_mu).pow(2)) / (2 * self.prior_weight_sigma.pow(2))
                      - 0.5).sum()

        kl_bias = (torch.log(self.prior_bias_sigma / posterior_bias_sigma)
                   + (posterior_bias_sigma.pow(2) + (self.bias_mu - self.prior_bias_mu).pow(2)) / (2 * self.prior_bias_sigma.pow(2))
                   - 0.5).sum()

        return kl_weights + kl_bias

    def forward(self, x):
        """
        Performs a forward pass using a single sample from the posterior.
        """
        # 1. Get sigma from rho
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        # 2. Sample epsilon (noise) from N(0, I)
        # randn_like creates noise on the same device as the parameter
        epsilon_weight = torch.randn_like(weight_sigma)
        epsilon_bias = torch.randn_like(bias_sigma)

        # 3. Apply the reparameterization trick to sample weights and biases
        weight = self.weight_mu + epsilon_weight * weight_sigma
        bias = self.bias_mu + epsilon_bias * bias_sigma

        # 4. Perform the linear operation
        return F.linear(x, weight, bias)