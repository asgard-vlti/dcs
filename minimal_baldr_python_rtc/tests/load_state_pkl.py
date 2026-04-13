# %%
import pickle
import numpy as np
import pathlib
import glob

# %%
pth = pathlib.Path("~/tmp/baldr_minimal_py/").expanduser()

beam = 1

files = sorted(glob.glob(str(pth / f"bao_state_*.pkl")))

last_file = files[-1]

with open(last_file, "rb") as f:
    state = pickle.load(f)

# %%
state["recon"]
