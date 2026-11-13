# Installing the Verilator harness on macOS

For macOS 12 or later, on Apple Silicon (M1–M4) or Intel. About 15 minutes, most of it Homebrew.

Vivado does not run on macOS, so the Mac runs only the Verilator harness. Your xsim setup stays on
Windows; step 9 shows how to compare the two.

## 1. Command-line tools

```bash
xcode-select --install
```

If it says they are already installed, continue. This provides `clang++`, `make` and `git`.

## 2. Homebrew

If `brew --version` fails, install Homebrew from https://brew.sh and follow the two `eval` lines it
prints at the end, so `brew` is on your `PATH`.

## 3. Verilator

```bash
brew install verilator
verilator --version          # must print 5.x
```

Homebrew's Verilator is configured for Apple's `clang++`, which the Makefile picks up automatically.

## 4. Python

Use the Python you train with. A virtual environment keeps the build and the import in step:

```bash
cd ~/your-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pybind11 numpy
```

If you use conda, activate the conda env instead and run the same `pip install`.

**Check the architecture.** On Apple Silicon, Python, Homebrew and the terminal must all be `arm64`:

```bash
uname -m                                                 # arm64
python -c "import platform; print(platform.machine())"   # arm64
file "$(which verilator)"                                # ... arm64
```

If any of them says `x86_64` on an Apple Silicon Mac, that program runs under Rosetta and the module
will not import. The usual cause is a terminal opened with "Open using Rosetta" or an Intel-only
Python installer. Fix that before building. (On an Intel Mac all three say `x86_64`, which is fine.)

## 5. Get the package next to your RTL

```bash
unzip verilator_harness.zip -d ~/your-project/
cd ~/your-project/verilator_harness
```

You need the `rtl/new_decomp_v20_from_v19/clean/` directory from your T8_Decomp_xsim checkout on
the Mac too (copy it, or clone the repo). It must be the same RTL your xsim flow uses.

## 6. Check the toolchain

```bash
make doctor PYTHON="$(which python)" RTL_DIR=/path/to/rtl/new_decomp_v20_from_v19/clean
```

Expected, roughly:

```
OS:          Darwin arm64
compiler:    clang++ -> Apple clang version 1x.x.x
  verilator was configured with: clang++
verilator:   Verilator 5.xxx
python:      /Users/you/your-project/.venv/bin/python 3.12.x arm64
pybind11:    2.x or 3.x
module name: decomp_sim.cpython-312-darwin.so
RTL_DIR:     /path/to/.../clean (16 .v files)
```

Anything that says `NOT FOUND`, or `0 .v files`, needs fixing first.

## 7. Build

```bash
make PYTHON="$(which python)" RTL_DIR=/path/to/rtl/new_decomp_v20_from_v19/clean
```

The first build Verilates and compiles the RTL (a minute or two), then links
`python/decomp_sim.cpython-3xx-darwin.so`.

Quick check:

```bash
cd python
python -c "import decomp_sim as d; s = d.DecompSim(); s.reset(); [s.step_idle() for _ in range(1000)]; print('ok, cycle', s.cycle)"
cd ..
```

It should print `ok, cycle 1001`.

## 8. Run the tests

You need the data files (`<stem>_subdec<s>_beats.hex` and `_rows.hex`) and `tb_interactive.sv` on
the Mac.

```bash
make reference PYTHON="$(which python)" RTL_DIR=/path/to/clean TB=/path/to/sim/tb_interactive.sv
cd tests
python test_equivalence.py --ref ../ref_obj/tb_ref --stem /path/to/4_comp_data/test_11_lantern
python test_client.py --stem /path/to/4_comp_data/test_11_lantern
```

Both end with `PASS`. The equivalence test prints one `%Error: ... Assertion failed` line on
purpose: its `stop` case sends an impossible beat to check that the harness stops on the same
cycle as the reference.

## 9. Compare with real xsim (record on Windows, replay on the Mac)

On the Windows machine, wrap the xsim client once and run one or two episodes (README, "Still to
verify against real xsim"). Copy the resulting `xsim_session.log` to the Mac, then:

```bash
cd tests
python xsim_replay.py /path/to/xsim_session.log
```

`PASS` means the harness matches xsim cycle for cycle. If it fails a few hundred cycles in, try
`--next-edge` (the reset-release question explained in the README) and use whichever passes.

The log stores the image paths the Windows env used (lines starting with `A ` right after each
`#start` / `#restart`). Before replaying, make the same data files available on the Mac and
rewrite those paths. They may contain backslashes, because `env.reset()` passes the path as
Windows formats it. Converting them with a short Python snippet is the easiest way:

```bash
python - <<'EOF2'
import re
src, dst = r"C:\Users\you\project\data", "/Users/you/your-project/data"
log = open("xsim_session.log").read()
log = log.replace(src, dst)
log = re.sub(r"^(A /\S*)$", lambda m: m.group(1).replace("\\", "/"), log, flags=re.M)
open("xsim_session.log", "w").write(log)
EOF2
```

Paths with spaces are not supported (the testbench reads the path with `%s` too).

## 10. Use it in the env

Copy `python/verilator_sim_client.py` and `python/decomp_sim.cpython-3xx-darwin.so` into the env's
package directory (next to `env.py`), then change one import in `env.py`:

```python
from .verilator_sim_client import VerilatorSimClient as XSimClient
```

and turn RTL assertions into a failure terminal. `RtlStop` can come from any simulated cycle,
so catch it around the whole body of `step()` (both `self._tick(...)` and
`self._skip_until_ready()` advance the clock):

```python
import decomp_sim   # or: from . import decomp_sim

def step(self, action):
    try:
        return self._step(action)          # the current body of step(), renamed
    except decomp_sim.RtlStop as e:
        print(f"[Env] RTL stopped: {e}")
        return self._fail(violation="rtl_stop", slots_used=0)
```

After an `RtlStop` the client refuses to step until the next `reset()` (which calls `restart()`),
exactly like a stopped simulator.

The `.so` only works with the Python version and architecture it was built for. Rebuild it on each
machine (`make clean && make ...`) rather than copying it between computers.

## Troubleshooting

**`ld: warning: -undefined dynamic_lookup may not work with chained fixups`.** Harmless. Every
pybind11/setuptools extension on recent Xcode prints it.

**`ImportError: ... incompatible architecture (have 'x86_64', need 'arm64')`** (or the reverse).
The module and the Python importing it differ in architecture. Redo the checks in step 4, then
`make clean` and build again with the right `PYTHON`.

**`ImportError: ... Symbol not found: _Py...` or `No module named decomp_sim`.** Built with one
Python, imported with another. `make clean`, then build with `PYTHON="$(which python)"` from the
environment you train in.

**`pybind11 not importable by python3`.** `make` is using a different Python than the one you
installed `pybind11` into. Pass `PYTHON="$(which python)"`.

**Verilator's build step fails inside its own Makefile.** macOS ships GNU Make 3.81 from 2006.
`brew install make`, then run `gmake` instead of `make` for every command above.

**Compile errors mentioning C++ standard features in `verilated*.h`.** A newer Verilator may need a
newer standard: add `CXXSTD=c++20`.

**`make reference` fails with coroutine errors.** The reference is built with Verilator's own
compiler settings. Don't pass `CXX=` for this target, and make sure `brew upgrade verilator`
and the Xcode command-line tools are current.

**Anything else after switching Python, compiler or Verilator version.** `make clean` first: the
Verilated model in `obj_dir/` is reused across builds.
