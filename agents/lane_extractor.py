"""SB3 features extractor for the v2 per-lane observation (envs/xsim/observation_v2.py)."""
import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LaneSharedExtractor(BaseFeaturesExtractor):
    """Shared per-lane MLP.

    Each lane's features, concatenated with the globals, go through the same MLP, so what is
    learnt about one lane transfers to all of them. The lane embeddings are then concatenated in
    lane order (the action head still needs to know which lane is which), followed by the globals.

    Usage: policy_kwargs=dict(features_extractor_class=LaneSharedExtractor,
                              features_extractor_kwargs=dict(lane_embed_dim=32))
    """

    def __init__(self, observation_space: gym.spaces.Dict, lane_embed_dim: int = 32, hidden_dim: int = 64):
        n_lanes, n_lane_features = observation_space["lanes"].shape
        n_globals = observation_space["globals"].shape[0]
        super().__init__(observation_space, features_dim=n_lanes * lane_embed_dim + n_globals)

        self.lane_mlp = nn.Sequential(
            nn.Linear(n_lane_features + n_globals, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, lane_embed_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> th.Tensor:
        lanes = observations["lanes"]                      # (batch, n_lanes, n_lane_features)
        globals_ = observations["globals"]                 # (batch, n_globals)
        per_lane_globals = globals_.unsqueeze(1).expand(-1, lanes.shape[1], -1)
        lane_embeddings = self.lane_mlp(th.cat([lanes, per_lane_globals], dim=-1))
        return th.cat([lane_embeddings.flatten(start_dim=1), globals_], dim=1)
