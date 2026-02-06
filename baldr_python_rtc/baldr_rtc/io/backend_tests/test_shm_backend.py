# baldr_rtc/io/backend_tests/test_shm_backend.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from baldr_python_rtc.baldr_rtc.io import shm_backend


beam=3
cam_path = None#f"/dev/shm/baldr{beam}.im.shm"
cam = shm_backend.ShmCameraIO(beam = 3, nosem=False, semid=3)

fr = cam.get_frame()

assert hasattr(fr, "data")
assert hasattr(fr, "t_s")
assert hasattr(fr, "frame_id")



# look on camera 
test_cmd  = np.zeros(140)
dm.set_data( test_cmd  )
input('set zero on channel 2 dm, press anything to put another shape on ')
test_cmd[64] = 0.2
dm.set_data( test_cmd  )
input('returned to zeros on channel 2 , press anything to put another shape on ')
test_cmd  = np.zeros(140)
dm.set_data( test_cmd  )
