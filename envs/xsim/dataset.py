"""Loading of per-subdecoder hex command streams."""
from collections import deque
from pathlib import Path

from .constants import NUM_DECODERS


def load_hex_folder(cmd_path: Path) -> list:
    files = sorted(cmd_path.glob('*.hex'))
    if len(files) < NUM_DECODERS:
        raise ValueError(f"Expected {NUM_DECODERS} subdecoder hex files, found {len(files)}")

    outputs = []
    for file_path in files[:NUM_DECODERS]:
        raw_lines = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]
        outputs.append(deque(int(ln, 16) for ln in raw_lines))
    return outputs


def _stem_path(hex_dir: Path) -> Path:
    first_file = next(hex_dir.glob('*.hex')).name
    return hex_dir / first_file.split('_subdec')[0]


def load_dataset_pool(data_dir: Path, is_inference: bool = False) -> list:
    """Return a list of (stem_path, cmds) samples.

    Inference: data_dir is a single beats_hex folder.
    Training: data_dir holds one sub-folder per sample, each with a beats_hex folder.
    """
    if is_inference:
        cmds = load_hex_folder(data_dir)
        stem_path = _stem_path(data_dir)
        print(f"Loaded single dataset for inference from: {data_dir}")
        return [(stem_path, cmds)]

    all_datasets = []
    for test_folder in sorted(data_dir.iterdir()):
        if test_folder.is_dir():
            hex_dir = test_folder / "beats_hex"
            if hex_dir.exists() and hex_dir.is_dir():
                try:
                    cmds = load_hex_folder(hex_dir)
                    all_datasets.append((_stem_path(hex_dir), cmds))
                except Exception as e:
                    print(f"Skipping {test_folder.name} due to error: {e}")

    if not all_datasets:
        raise ValueError(f"No valid datasets found in {data_dir}")
    return all_datasets
