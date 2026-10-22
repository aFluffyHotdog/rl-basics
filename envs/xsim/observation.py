"""Observation space and observation construction (live hardware state + input lookahead)."""
import gymnasium as gym
import numpy as np

from .constants import (
    CMD_FIFO_SIZE,
    LIT_FIELDS,
    LIT_FIFO_CAPACITIES,
    LOOKAHEAD_WINDOW,
    MAX_CYCLES_LEFT,
    NUM_ACTIONS,
    NUM_DECODERS,
    OUT_FIFO_SIZE,
)
from .flow_control import cmd_inflight
from .tokens import TYPE_INDEX, decode_token


def build_observation_space() -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "top_in_ready": gym.spaces.Discrete(2),
        "top_row_valid": gym.spaces.Discrete(2),
        "sub_decoder_fifo_counts": gym.spaces.Box(
            low=0.0, high=1.0, shape=(NUM_DECODERS, 6), dtype=np.float32
        ),
        "cmd_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
        "lit_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
        "output_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
        "lookahead_types": gym.spaces.Box(
            low=0,
            high=1,
            shape=(NUM_DECODERS, LOOKAHEAD_WINDOW, 4),
            dtype=np.int32,
        ),
        "global_progress": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        "cycles_left": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
        "lookahead_cycles": gym.spaces.Box(low=-1.0, high=1.0, shape=(NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.float32),
        "lookahead_is_barrier": gym.spaces.Box(low=-1.0, high=1.0, shape=(NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.int32),
        "channel_skew": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
        # Why a lane is (un)available .
        "tail_busy": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
        "cmd_inflight": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
        "action_mask": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_ACTIONS,), dtype=np.float32),
    })


def _empty_observation(hw_state: dict) -> dict:
    return {
        "top_in_ready": np.array(hw_state["top_in_ready"], dtype=np.int64),
        "top_row_valid": np.array(hw_state["top_row_valid"], dtype=np.int64),
        "sub_decoder_fifo_counts": np.zeros((NUM_DECODERS, 6), dtype=np.float32),
        "cmd_fifo_fill": np.zeros(NUM_DECODERS, dtype=np.float32),
        "lit_fifo_fill": np.zeros(NUM_DECODERS, dtype=np.float32),
        "output_fifo_fill": np.zeros(NUM_DECODERS, dtype=np.float32),
        "lookahead_types": np.zeros((NUM_DECODERS, LOOKAHEAD_WINDOW, 4), dtype=np.int32),
        "global_progress": np.array([0.0], dtype=np.float32),
        "cycles_left": np.zeros(NUM_DECODERS, dtype=np.float32),
        "lookahead_cycles": np.full((NUM_DECODERS, LOOKAHEAD_WINDOW), -1, dtype=np.float32),
        "lookahead_is_barrier": np.full((NUM_DECODERS, LOOKAHEAD_WINDOW), -1, dtype=np.int32),
        "channel_skew": np.zeros(NUM_DECODERS, dtype=np.float32),
        "tail_busy": np.zeros(NUM_DECODERS, dtype=np.float32),
        "cmd_inflight": np.zeros(NUM_DECODERS, dtype=np.float32),
        "action_mask": np.zeros(NUM_ACTIONS, dtype=np.float32),
    }


def _lookahead_features(obs: dict, cmds: list):
    for i in range(NUM_DECODERS):
        bus_queue = cmds[i]

        if len(bus_queue) > 0:
            obs["cycles_left"][i] = min(decode_token(bus_queue[0]).cycles / MAX_CYCLES_LEFT, 1.0)
        else:
            obs["cycles_left"][i] = 0.0

        window_size = min(len(bus_queue), LOOKAHEAD_WINDOW)
        for j in range(window_size):
            token_info = decode_token(bus_queue[j])
            obs["lookahead_cycles"][i, j] = min(token_info.cycles / MAX_CYCLES_LEFT, 1.0)
            obs["lookahead_is_barrier"][i, j] = float(token_info.is_barrier)
            obs["lookahead_types"][i, j, TYPE_INDEX[token_info.category]] = 1


def _fifo_features(obs: dict, hw_state: dict):
    for i, decoder in enumerate(hw_state.get("sub_decoders", [])[:NUM_DECODERS]):
        lit_fills = [decoder[field] / cap for field, cap in zip(LIT_FIELDS, LIT_FIFO_CAPACITIES)]
        obs["sub_decoder_fifo_counts"][i] = [
            decoder["cmd"] / CMD_FIFO_SIZE,
            *lit_fills,
            decoder["out"] / OUT_FIFO_SIZE,
        ]
        obs["cmd_fifo_fill"][i] = decoder["cmd"] / CMD_FIFO_SIZE
        obs["lit_fifo_fill"][i] = max(lit_fills)
        obs["output_fifo_fill"][i] = decoder["out"] / OUT_FIFO_SIZE


def _flow_control_features(obs: dict, hw_state: dict, n_sent, action_mask):
    # Flow-control state behind the mask. tail_busy needs the extended testbench (zeros otherwise);
    # in-flight falls back to FIFO occupancy on the older testbench. Normalised by the CMD FIFO depth.
    tail_busy = hw_state.get("tail_busy")
    if tail_busy is not None:
        obs["tail_busy"][:] = tail_busy
    if "sub_decoders" in hw_state and n_sent is not None:
        for i in range(NUM_DECODERS):
            obs["cmd_inflight"][i] = min(max(cmd_inflight(hw_state, n_sent, i), 0) / CMD_FIFO_SIZE, 1.0)
        obs["action_mask"][:] = action_mask()   # mask for the action about to be chosen


def build_observation(hw_state: dict, cmds: list, initial_tokens: int, n_sent, action_mask) -> dict:
    """action_mask is a callable so the (cached) mask is only computed when it is observable."""
    obs = _empty_observation(hw_state)
    _lookahead_features(obs, cmds)
    _fifo_features(obs, hw_state)

    total_remaining = sum(len(q) for q in cmds)
    obs["global_progress"][0] = total_remaining / max(1, initial_tokens)

    _flow_control_features(obs, hw_state, n_sent, action_mask)
    return obs
