from pathlib import Path
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import minimal_baldr_python_rtc.DM as DM
import minimal_baldr_python_rtc.utils as utils
import minimal_baldr_python_rtc.Cam as Cam

import numpy as np

beams = [1, 2, 3, 4]
DMs = [DM.FourierDM(i) for i in beams]
cams = [Cam.Cam(i) for i in beams]


def run_ramp_single(
    beam,
    cam,
    dm,
    max_amp=0.5,
    n_steps=100,
    sleep=0.01,
    n_discard=1,
    n_im=30,
):
    ctx, sock = utils.mds_connect("mimir", 5555)
    ramp_amps = np.linspace(-max_amp, max_amp, n_steps)

    try:
        dm.flatten()

        bmy_pos = utils.mds_send(sock, f"read BMY{beam}")
        utils.mds_send(sock, f"moveabs BMY{beam} 500.0")
        time.sleep(3)

        cam.take_dark(256)

        dark = cam.take_stack(1000)

        utils.mds_send(sock, f"moveabs BMY{beam} {bmy_pos}")
        time.sleep(3)

        # take pupil only image
        offset = 200.0
        utils.mds_send(sock, f"moverel BMY{beam} {-offset}")
        time.sleep(1)

        pupil_only = cam.take_stack(1000)

        utils.mds_send(sock, f"moverel BMY{beam} {offset}")
        time.sleep(1)


        ref = cam.take_stack(1000)

        # apply ramp
        ims = []
        for mode_idx in range(dm.n_acts):
            im_mode = []
            for amp in ramp_amps:
                cmd = np.zeros(dm.n_acts)
                cmd[mode_idx] = amp
                dm.set_data(cmd)

                time.sleep(sleep)

                for _ in range(n_discard):
                    cam.get_img()

                im = cam.take_stack(n_im).mean(0)

                im_mode.append(im)
            ims.append(im_mode)

        dm.flatten()

        return np.array(ims), pupil_only, ramp_amps, dark, ref
    finally:
        sock.close()
        ctx.term()


def run_and_save_single(beam, cam, dm, out_root):
    ims, pupil_only, ramp_amps, dark, ref = run_ramp_single(beam=beam, cam=cam, dm=dm)

    beam_dir = out_root / f"beam{beam}"
    beam_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = beam_dir / f"{timestamp}.npz"

    np.savez_compressed(
        out_file,
        beam=beam,
        ims=ims,
        pupil_only=pupil_only,
        ramp_amps=ramp_amps,
        dark=dark,
        ref=ref,
    )
    return out_file


def main():
    out_root = Path.home() / "tmp" / "minimal_baldr_rtc" / "fourier_ramps"

    futures = {}
    with ThreadPoolExecutor(max_workers=len(beams)) as executor:
        for beam, cam, dm in zip(beams, cams, DMs):
            future = executor.submit(run_and_save_single, beam, cam, dm, out_root)
            print(f"submitted beam {beam} to start")
            futures[future] = beam

        for future in as_completed(futures):
            beam = futures[future]
            out_file = future.result()
            print(f"Beam {beam} saved to {out_file}")


if __name__ == "__main__":
    main()
