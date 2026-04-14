# %%
import subprocess
import minimal_baldr_python_rtc.DM as DM
import consts
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

beam = 1


dm = DM.DM(beam)

subprocess.run(["flat-load", str(beam), "night-standard"], check=True)
print(f"Loaded night standard for beam {beam}")


flat = dm.shms[0].get_data().copy()

plt.imshow(flat.reshape(consts.act_shape))

# %%
# lowpass filter, masking the corners to avoid edge effects
flat = flat.reshape(consts.act_shape)

flat[0, 0] = flat[0, 1]
flat[0, -1] = flat[0, -2]
flat[-1, 0] = flat[-2, 0]
flat[-1, -1] = flat[-2, -2]

filtered_flat = gaussian_filter(flat, sigma=1.0)

# set corners back to zero
filtered_flat[0, 0] = 0.0
filtered_flat[0, -1] = 0.0
filtered_flat[-1, 0] = 0.0
filtered_flat[-1, -1] = 0.0

plt.imshow(filtered_flat)


# %%
dm.set_data(filtered_flat.flatten(), chn=0)
