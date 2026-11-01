from pathlib import Path

from deployment.replay_beats_xsim import load_beats


def test_load_beats_ignores_idle_lines_and_comments(tmp_path, capsys):
    beat = f"{(2 << 131) | 1:034x}"
    beats_path = tmp_path / "beats.hex"
    beats_path.write_text(
        f"{beat} // DATA cycle=1 ready=1\n"
        "invalid // IDLE cycle=2 ready=0\n"
        "// comment-only line\n"
        f"1 {beat} # action with comment\n",
        encoding="ascii",
    )

    assert load_beats(beats_path) == [beat, beat]
    assert "'IDLE': 1" in capsys.readouterr().out


def test_load_beats_ignores_hash_comment_only_lines(tmp_path):
    beats_path = tmp_path / "beats.hex"
    beat = f"{(2 << 131) | 1:034x}"
    beats_path.write_text(f"# comment-only line\n{beat}\n", encoding="ascii")

    assert load_beats(beats_path) == [beat]
