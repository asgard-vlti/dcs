#!/usr/bin/env python3

from xaosim.shmlib import shm
import numpy as np

dmid = 1
dms = 12
zeros = np.zeros((dms, dms))

shm1 = shm(f"/dev/shm/dm{dmid}disp.im.shm", nosem=False)
shm_data = shm(f"/dev/shm/dm{dmid}disp03.im.shm", nosem=False)

for ii in range(50):
    shm_data.set_data(zeros+ii)
    shm1.post_sems(1)

# shm1.close(erase_file=False)
