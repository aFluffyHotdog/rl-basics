#!/usr/bin/env python3
"""Pack compressed data into 135b beats and send it to the board over UDP.

This version of the script is "Channel-Major" (Sequential). It does not 
interleave the channels. It sends all beats for Channel 0, then all beats 
for Channel 1, etc. 

You can also isolate a single subdecoder using the `--subdec` argument.

Reads from a single directory containing subdec hex files:
  subdec0_beats.hex .. subdec7_beats.hex   (regular sub-decs, CMD/LIT/RLE chunks)
  subdec8_rows.hex                          (bypass, raw 56b rows)

Usage:
    udp_send_beats.py <hex_dir> [--delay SECONDS] [--timeout SECONDS] [--subdec CHANNEL] [--schedule <schedule.txt path>]

--subdec:   Only process and send the 135b beats for this specific channel (0-8).
--delay:    Adds a pause between every packet sent.
--timeout:  Controls how long to wait for the board's cycle-count reply.
"""
import glob
import os
import socket
import struct
import sys
import time

TYPE_BY_NIBBLE = {0: "CMD", 1: "LIT", 2: "RLE"}
NUM_REGULAR_SUBDECS = 8
BYPASS_DEST = 8
BYTES_PER_BEAT = 20
BOARD_IP = "192.168.1.10"
BOARD_PORT = 5001
PC_REPLY_PORT = 5002       # must match PC_REPLY_PORT in main.c
RECV_TIMEOUT_S = 5.0
OUT_BEATS_FILENAME = "beats_sent.txt"
CTRL_MARKER_BEGIN = 0x00
CTRL_MARKER_END = 0x01


def find_one(hex_dir, pattern):
    matches = sorted(glob.glob(os.path.join(hex_dir, pattern)))
    if len(matches) == 0:
        raise ValueError("{}: no file matching {!r} found".format(hex_dir, pattern))
    if len(matches) > 1:
        raise ValueError("{}: multiple files matching {!r} found: {} "
                          "(directory must contain exactly one set of subdec "
                          "files)".format(hex_dir, pattern, matches))
    return matches[0]


def parse_chunks(path):
    chunks = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            if len(s) != 5:
                raise ValueError("{}:{}: expected 5 hex chars, got {!r}".format(path, ln, s))
            typ = TYPE_BY_NIBBLE.get(int(s[0], 16))
            if typ is None:
                raise ValueError("{}:{}: bad type nibble in {!r}".format(path, ln, s))
            chunks.append((typ, int(s[1:], 16)))
    return chunks


def group_chunks(chunks, path):
    """CMD + the chunks its own bits say it consumes -> one group."""
    groups = []
    i, n = 0, len(chunks)
    while i < n:
        typ, val = chunks[i]
        if typ != "CMD":
            raise ValueError("{}: expected CMD at chunk {}, got {}".format(path, i, typ))
        is_rle = (val >> 15) & 1
        if is_rle:
            k, want = 1, "RLE"
        else:
            k, want = (val >> 11) & 0xF, "LIT"
        if 1 + k > 8:
            raise ValueError("{}: CMD at chunk {} needs {} slots (1 + {} {}), "
                              "exceeds one 8-slot beat".format(path, i, 1 + k, k, want))
        group = [val]
        for j in range(k):
            idx = i + 1 + j
            if idx >= n:
                raise ValueError("{}: truncated stream: CMD at chunk {} needs {} "
                                  "{} chunks, ran out at {}".format(path, i, k, want, idx))
            t2, v2 = chunks[idx]
            if t2 != want:
                raise ValueError("{}: expected {} at chunk {}, got {}".format(path, want, idx, t2))
            group.append(v2)
        groups.append(group)
        i += 1 + k
    return groups


def pack_beats(groups):
    """Greedily pack consecutive groups into <=8-slot beats."""
    beats = []
    slots, ngroups = [], 0
    for gi, g in enumerate(groups):
        if len(slots) + len(g) > 8:
            beats.append((slots, ngroups))
            slots, ngroups = [], 0
        slots.extend(g)
        ngroups += 1
    if slots:
        beats.append((slots, ngroups))
    return beats


def beat_value(dest, ngroups, slots):
    payload = 0
    for i, w in enumerate(slots):
        payload |= (w & 0xFFFF) << (16 * i)
    return ((dest & 0xF) << 131) | (((ngroups - 1) & 0x7) << 128) | payload


def parse_bypass_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(int(s, 16))
    return rows


def pack_bypass_beats(rows, path):
    if len(rows) % 2 != 0:
        raise ValueError("{}: odd row count {}, bypass packs 2 rows/beat".format(path, len(rows)))
    out = []
    for bi, k in enumerate(range(0, len(rows), 2)):
        lo, hi = rows[k], rows[k + 1]
        payload = ((hi & ((1 << 64) - 1)) << 64) | (lo & ((1 << 64) - 1))
        out.append((bi, (BYPASS_DEST << 131) | payload))
    return out


def parse_schedule(schedule_path):
    """Parses a text file of subdecoder IDs (supports newlines or commas)."""
    schedule = []
    with open(schedule_path, 'r') as f:
        # Replace commas with newlines to support both CSV and vertical lists
        content = f.read().replace(',', '\n')
        for line in content.split():
            s = line.strip()
            if s:
                schedule.append(int(s))
    return schedule


def build_scheduled_beats(hex_dir, schedule_path):
    """Returns a flat list of 135b beat ints ordered by the provided schedule txt file."""
    queues = {i: [] for i in range(9)}
    
    # Build queues for regular subdecoders (0-7)
    for dest in range(NUM_REGULAR_SUBDECS):
        path = find_one(hex_dir, "*subdec{}_beats.hex".format(dest))
        chunks = parse_chunks(path)
        groups = group_chunks(chunks, path)
        beats = pack_beats(groups)
        
        for slots, ngroups in beats:
            queues[dest].append(beat_value(dest, ngroups, slots))
            
        print("{}: {} chunks -> {} groups -> {} beats".format(path, len(chunks), len(groups), len(beats)))

    # Build queue for Bypass Channel (8)
    bpath = find_one(hex_dir, "*subdec8_rows.hex")
    brows = parse_bypass_rows(bpath)
    bypass_beats = pack_bypass_beats(brows, bpath)
    
    for _, val in bypass_beats:
        queues[BYPASS_DEST].append(val)
        
    print("{}: {} rows -> {} beats".format(bpath, len(brows), len(bypass_beats)))

    # Apply the Schedule
    schedule = parse_schedule(schedule_path)
    final_beats = []
    
    for action_idx, dest in enumerate(schedule):
        if dest in queues and queues[dest]:
            final_beats.append(queues[dest].pop(0))
            
    # Safety net: Handle leftover beats that weren't covered by the schedule
    leftovers = sum(len(q) for q in queues.values())
    if leftovers > 0:
        print("\n[WARNING] Schedule exhausted, but {} beats remain in queues!".format(leftovers))
        print("Appending the remaining beats sequentially to prevent image corruption.")
        for dest in range(9):
            while queues[dest]:
                final_beats.append(queues[dest].pop(0))
                
    return final_beats


def beat_to_words(beat):
    """135b beat -> 5 x 32-bit words per dec_axis_unpack's layout."""
    word0 = beat & 0xFFFFFFFF
    word1 = (beat >> 32) & 0xFFFFFFFF
    word2 = (beat >> 64) & 0xFFFFFFFF
    word3 = (beat >> 96) & 0xFFFFFFFF
    tail7 = (beat >> 128) & 0x7F   # dest[134:131] + num_cmd_m1[130:128]
    word4 = tail7                  # upper 25 bits zero by construction
    return word0, word1, word2, word3, word4


def pack_beat_bytes(beat):
    w0, w1, w2, w3, w4 = beat_to_words(beat)
    # little-endian 32-bit words, matching a 32-bit AXIS width sender
    return struct.pack("<IIIII", w0, w1, w2, w3, w4)


def send_beats_streamed(beats, ip, port, delay_s=0.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(bytes([CTRL_MARKER_BEGIN]), (ip, port))
        if delay_s:
            time.sleep(delay_s)

        for beat in beats:
            sock.sendto(pack_beat_bytes(beat), (ip, port))
            if delay_s:
                time.sleep(delay_s)

        sock.sendto(bytes([CTRL_MARKER_END]), (ip, port))
    finally:
        sock.close()


def recv_cycle_count(recv_sock, timeout_s=RECV_TIMEOUT_S):
    recv_sock.settimeout(timeout_s)
    try:
        data, addr = recv_sock.recvfrom(1024)
    except socket.timeout:
        print("no reply from board within {}s (timeout waiting for cycle count)"
              .format(timeout_s))
        return None

    if len(data) != 4:
        print("warning: reply from {} was {} bytes, expected 4 -- ignoring"
              .format(addr, len(data)))
        return None

    (cycles,) = struct.unpack("!I", data)
    print("reply from {}: cycle_count={}".format(addr, cycles))
    return cycles


def main():
    args = sys.argv[1:]
    delay_s = 0.0
    timeout_s = RECV_TIMEOUT_S
    schedule_path = None

    if "--delay" in args:
        i = args.index("--delay")
        delay_s = float(args[i + 1])
        del args[i:i + 2]

    if "--timeout" in args:
        i = args.index("--timeout")
        timeout_s = float(args[i + 1])
        del args[i:i + 2]
        
    if "--schedule" in args:
        i = args.index("--schedule")
        schedule_path = args[i + 1]
        del args[i:i + 2]

    if len(args) != 1 or not schedule_path:
        sys.stderr.write("usage: udp_send_beats.py <hex_dir> --schedule <schedule.txt> [--delay SECONDS] [--timeout SECONDS]\n")
        sys.exit(1)

    hex_dir = args[0]
    ip = BOARD_IP
    port = BOARD_PORT

    beats = build_scheduled_beats(hex_dir, schedule_path)
    
    if not beats:
        sys.stderr.write("no beats produced from {}\n".format(hex_dir))
        sys.exit(1)
        
    print("total {} beats (Scheduled via {})".format(len(beats), os.path.basename(schedule_path)))

    out_path = os.path.join(hex_dir, OUT_BEATS_FILENAME)
    with open(out_path, "w") as f:
        for b in beats:
            f.write("{:034x}\n".format(b))
    print("wrote beat stream to {} (for verification against what was sent)".format(out_path))

    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    recv_sock.bind(("0.0.0.0", PC_REPLY_PORT))

    try:
        send_beats_streamed(beats, ip, port, delay_s=delay_s)
        print("sent BEGIN + {} beats (one datagram each) + END to {}:{}"
              .format(len(beats), ip, port))

        cycles = recv_cycle_count(recv_sock, timeout_s=timeout_s)
        if cycles is not None:
            print("episode cycle_count = {}".format(cycles))
    finally:
        recv_sock.close()


if __name__ == "__main__":
    main()