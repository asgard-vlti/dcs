from pathlib import Path
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import minimal_baldr_python_rtc.DM as DM
import minimal_baldr_python_rtc.utils as utils
import minimal_baldr_python_rtc.Cam as Cam

import numpy as np
from tqdm.auto import tqdm

beams = [1, 2, 3, 4]
DMs = [DM.FourierDM(i) for i in beams]
cams = [Cam.Cam(i) for i in beams]


_TQDM_LOCK = threading.RLock()
tqdm.set_lock(_TQDM_LOCK)


def run_ramp_single(
    beam,
    cam,
    dm,
    beam_position,
    max_amp=0.7,
    n_steps=50,
    sleep=0.01,
    n_discard=1,
    n_im=100,
):
    ctx, sock = utils.mds_connect("mimir", 5555)
    ramp_amps = np.linspace(-max_amp, max_amp, n_steps)

    try:
        dm.flatten()

        bmy_pos = utils.mds_send(sock, f"read BMY{beam}")
        utils.mds_send(sock, f"moveabs BMY{beam} 500.0")
        time.sleep(3)

        cam.take_dark(256)

        cam_dark = cam.dark.copy()
        dark = cam.take_stack(1000)

        utils.mds_send(sock, f"moveabs BMY{beam} {bmy_pos}")
        time.sleep(3)

        # take pupil only image
        offset = 200.0
        utils.mds_send(sock, f"moverel BMY{beam} {-offset}")
        time.sleep(1)

        pupil_only = cam.take_stack(2000)

        utils.mds_send(sock, f"moverel BMY{beam} {offset}")
        time.sleep(1)

        ref = cam.take_stack(2000)

        # apply ramp
        ims = []
        total_steps = dm.n_acts * len(ramp_amps)
        with tqdm(
            total=total_steps,
            desc=f"beam {beam}",
            position=beam_position,
            leave=True,
        ) as pbar:
            for mode_idx in range(dm.n_acts):
                im_mode = []
                for amp in ramp_amps:
                    cmd = np.zeros(dm.n_acts)
                    cmd[mode_idx] = amp
                    dm.set_data(cmd)

                    time.sleep(sleep)

                    for _ in range(n_discard):
                        cam.get_img()

                    im = cam.take_stack(n_im)

                    im_mode.append(im)
                    pbar.update(1)
                ims.append(im_mode)

        dm.flatten()

        return np.array(ims), pupil_only, ramp_amps, dark, ref, cam_dark
    finally:
        sock.close()
        ctx.term()


def run_and_save_single(beam, cam, dm, out_root, beam_position):
    beam_dir = out_root / f"beam{beam}"
    beam_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    ims, pupil_only, ramp_amps, dark, ref, cam_dark = run_ramp_single(
        beam=beam,
        cam=cam,
        dm=dm,
        beam_position=beam_position,
    )

    out_file = beam_dir / f"{timestamp}.npz"

    np.savez_compressed(
        out_file,
        beam=beam,
        ims=ims,
        pupil_only=pupil_only,
        ramp_amps=ramp_amps,
        dark=dark,
        ref=ref,
        cam_dark=cam_dark,
    )
    return out_file


def main():
    out_root =  Path("/data") / "AT" / "minimal_baldr_rtc" / "fourier_ramps"

    futures = {}
    with ThreadPoolExecutor(max_workers=len(beams)) as executor:
        for beam_position, (beam, cam, dm) in enumerate(zip(beams, cams, DMs)):
            future = executor.submit(
                run_and_save_single,
                beam,
                cam,
                dm,
                out_root,
                beam_position,
            )
            print(f"submitted beam {beam} to start")
            futures[future] = beam

        for future in as_completed(futures):
            beam = futures[future]
            out_file = future.result()
            print(f"Beam {beam} saved to {out_file}")


if __name__ == "__main__":
    main()
