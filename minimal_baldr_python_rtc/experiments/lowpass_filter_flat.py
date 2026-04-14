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
# lowpass filter
filtered_flat = gaussian_filter(flat.reshape(consts.act_shape), sigma=1.0)
plt.imshow(filtered_flat)
