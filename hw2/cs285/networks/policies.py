import itertools

import numpy as np
import torch
from torch import distributions, nn, optim
from torch.nn import functional as F

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        obs_t = ptu.from_numpy(obs)
        dist = self(obs_t)
        if deterministic:
            action_t = dist.mode
        else:
            action_t = dist.sample()
        action = ptu.to_numpy(action_t)
        return action

    def forward(self, obs: torch.Tensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            pass
            logits = self.logits_net(obs)
            dist = distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(obs)
            dist = distributions.Normal(mean, self.logstd.exp())
        return dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        obs_t = ptu.from_numpy(obs)
        actions_t = ptu.from_numpy(actions)
        dist = self.forward(obs_t)
        loss = -dist.log_prob(actions_t).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"Training Loss": ptu.to_numpy(loss)}


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs_t = ptu.from_numpy(obs)
        actions_t = ptu.from_numpy(actions)
        advantages_t = ptu.from_numpy(advantages)

        dist = self.forward(obs_t)
        loss = -(dist.log_prob(actions_t) * advantages_t).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
