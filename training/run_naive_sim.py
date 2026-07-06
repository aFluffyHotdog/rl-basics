#!/usr/bin/env python3
"""Naive runner: round-robin feed 128-bit beats to each sub-decoder and report cycles.

Usage: python training/run_naive_sim.py /path/to/beats_hex_folder
"""
from pathlib import Path
from collections import deque
import argparse, os, sys
from tqdm import tqdm

# Add parent directory to path so imports work from any location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.decoder_env_v2 import DecoderEnvV2, NUM_DECODERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd_path', nargs='?', default='sample_data/test_11_lantern/beats_hex')
    args = ap.parse_args()

    cmd_path = Path(args.cmd_path)
    if not cmd_path.exists():
        raise SystemExit(f"cmd_path not found: {cmd_path}")

    # instantiate environment (loads via its loader but we'll load tokens ourselves)
    env = DecoderEnvV2(str(cmd_path))
    # naive round-robin using env.step(action)
    # create a tqdm progress bar for subdecoder 0's token exhaustion
    initial0 = len(env.cmds[0]) if len(env.cmds) > 0 else 0
    pbar = tqdm(total=initial0, desc="Subdec0 tokens consumed") if initial0 > 0 else None
    prev_remaining = initial0

    terminated = False
    while not terminated:
        for d in range(NUM_DECODERS):
            _, _, terminated, _, _ = env.step(d)
            # update progress for subdecoder 0
            if pbar is not None:
                curr_remaining = len(env.cmds[0])
                consumed = prev_remaining - curr_remaining
                if consumed > 0:
                    pbar.update(consumed)
                    prev_remaining = curr_remaining
            if terminated:
                break

    # report
    print(f"Total beats (bus cycles) delivered: {env.num_cycles}")
    if pbar is not None:
        pbar.close()

    print("Per-decoder active time:")
    for i in range(NUM_DECODERS):
        print(f"  Decoder {i}: active_cycles_stat={env.decoders[i].active_cycles}")


if __name__ == '__main__':
    main()
