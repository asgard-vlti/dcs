"""
Autoalignment of Heimdallr beams.

Options to make correction in either
- image space or pupil space,
- using K1 or K2
- making corrections using MCS -> VLTI FSM, or internal tip-tilt mirrors.
"""

from xaosim.shmlib import shm
import numpy as np
import matplotlib.pyplot as plt
import time
import asgard_alignment.Engineering as asgE
import zmq
import os
from tqdm import tqdm
import scipy.ndimage as ndi

import dcs.ZMQutils

from scipy.optimize import curve_fit
import argparse

import json


class HeimdallrAA:
    def __init__(self, shutter_pause_time, band, flux_threshold, savepth, output, ncubes, t_pause):
        # Set target_pixels and col_bnds based on band
        centres = self.load_centre_positions()
        if band.upper() == "K1":
            self.target_pixels = centres["K1"]
            self.col_bnds = (0, 160)
        elif band.upper() == "K2":
            self.target_pixels = centres["K2"]
            self.col_bnds = (160, 320)
        else:
            raise ValueError("Unknown band: choose 'K1' or 'K2'.")

        self.flux_threshold = flux_threshold

        self.mds = self._open_mds_connection()

        self.stream = self._open_stream_connection()

        self.imsize = self._get_frame().shape

        self._shutter_pause_time = (
            shutter_pause_time  # seconds to pause after shuttering
        )

        self.ncubes = ncubes
        self.sleep_between_frames = t_pause  # seconds

        if savepth.lower() == "none":
            self.savepth = None
        else:
            self.savepth = savepth

        self.row_bnds = (1, centres["K1"][1] + 32)
        # self.col_bnds is set above

        if output not in ["internal", "mcs", "none"]:
            raise ValueError("Output must be 'internal' or 'mcs'")

        self.output = output

        if self.output == "mcs":
            self.mcs_client = dcs.ZMQutils.ZmqReq("tcp://192.168.100.2:7019")

    @staticmethod
    def load_centre_positions():
        # read in ~/.config/cred1_split.json
        config_path = os.path.expanduser("~/.config/cred1_split.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} does not exist.")
        with open(config_path, "r") as f:
            config = json.load(f)

        centre_pixels = {}
        x_c = config["hei_k1"]["x0"] + config["hei_k1"]["xsz"] // 2
        y_c = config["hei_k1"]["y0"] + config["hei_k1"]["ysz"] // 2
        centre_pixels["K1"] = (x_c, y_c)

        x_c = config["hei_k2"]["x0"] + config["hei_k2"]["xsz"] // 2
        y_c = config["hei_k2"]["y0"] + config["hei_k2"]["ysz"] // 2
        centre_pixels["K2"] = (x_c, y_c)

        return centre_pixels

    def get_init_motors_state(self):
        motors = {}
        for beam in range(1, 5):
            axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]
            axes = [axis + str(beam) for axis in axis_list]
            motors[beam] = {}
            for axis in axes:
                cmd = f"read {axis}"
                resp = self._send_and_get_response(cmd)
                motors[beam][axis] = float(resp)
        return motors

    def set_complete_state(self, motors):
        for beam in range(1, 5):
            for axis, pos in motors[beam].items():
                cmd = f"moveabs {axis} {pos}"
                self._send_and_get_response(cmd)

    # MDS interface
    def _open_mds_connection(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 10000)
        server_address = "tcp://192.168.100.2:5555"
        socket.connect(server_address)
        return socket

    def _send_and_get_response(self, message):
        # print("sending", message)
        self.mds.send_string(message)
        response = self.mds.recv_string()
        # print("response", response)
        return response.strip()

    # stream interface
    def _open_stream_connection(self):
        stream_path = "/dev/shm/cred1.im.shm"
        if not os.path.exists(stream_path):
            raise FileNotFoundError(f"Stream file {stream_path} does not exist.")
        return shm(stream_path)

    # processing
    def _find_blob_centre(self, frame):
        # use ndi median filter and then a gaussian filter on cropped image
        cropped_frame = frame[
            self.row_bnds[0] : self.row_bnds[1], self.col_bnds[0] : self.col_bnds[1]
        ]

        filtered_frame = ndi.median_filter(cropped_frame, size=3)
        filtered_frame = ndi.gaussian_filter(filtered_frame, sigma=2)

        max_loc_cropped = np.unravel_index(
            np.argmax(filtered_frame), filtered_frame.shape
        )
        max_loc_cropped = np.array(max_loc_cropped)
        max_loc = max_loc_cropped + np.array([self.row_bnds[0], self.col_bnds[0]])

        return max_loc

    def _get_frame(self):
        full_frame = self.stream.get_data().mean(0)

        # full_frame[ full_frame > np.quantile(full_frame, 0.98) ] = np.median( full_frame )
        return full_frame

    def _get_blob(self, radius=None, return_flux=False, inc_frame=False):
        """
        Find the final blob centre after frame averaging. Optionally measure flux in a circular region of given radius around the final blob centre.
        Args:
            radius (float or None): If not None and return_flux is True, measure flux in a circle of this radius around the final blob centre.
            return_flux (bool): Whether to return the flux measurement for the final blob.
            inc_frame (bool): If True and return_flux is True, also return the final frame used for flux measurement.
        Returns:
            If return_flux is False:
                blob_centre_final
            If return_flux is True and inc_frame is False:
                blob_centre_final, blob_flux
            If return_flux is True and inc_frame is True:
                blob_centre_final, blob_flux, acc_frame
        """
        acc_frame = np.zeros_like(self._get_frame())
        blob_centres = []
        for _ in range(self.ncubes):
            full_frame = self._get_frame() - 1000.0
            acc_frame += full_frame / self.ncubes
            blob_centre = self._find_blob_centre(full_frame)
            print("indiv blob centre", blob_centre)
            blob_centres.append(blob_centre)

            time.sleep(self.sleep_between_frames)
        # check the scatter on the blob centres
        blob_centres = np.array(blob_centres)
        scat = np.std(blob_centres, axis=0)
        if np.any(scat > 2.0):
            print(
                f"Warning: High scatter in blob centre positions: {scat}. Consider increasing ncubes or checking the image."
            )
        blob_centre_final = self._find_blob_centre(acc_frame)

        if not return_flux:
            return blob_centre_final
        if radius is None:
            raise ValueError("radius must be provided if return_flux is True")
        mask = self.circle_mask(acc_frame.shape, blob_centre_final, radius)
        blob_flux = acc_frame[mask].sum()
        if inc_frame:
            return blob_centre_final, blob_flux, acc_frame
        else:
            return blob_centre_final, blob_flux

    @staticmethod
    def circle_mask(shape, center, radius):
        y, x = np.ogrid[: shape[0], : shape[1]]
        dist_from_center = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        mask = dist_from_center <= radius
        return mask

    # def detect_pupil(self, image, sigma=2, threshold=0.5, plot=False, savepath=None):
    #     """
    #     Detects an elliptical pupil (with possible rotation) in a cropped image using edge detection
    #     and least-squares fitting. Returns both the ellipse parameters and a pupil mask.

    #     The ellipse is modeled by:

    #         ((x - cx)*cos(theta) + (y - cy)*sin(theta))^2 / a^2 +
    #         (-(x - cx)*sin(theta) + (y - cy)*cos(theta))^2 / b^2 = 1

    #     Parameters:
    #         image (2D array): Cropped grayscale image containing a single pupil.
    #         sigma (float): Standard deviation for Gaussian smoothing.
    #         threshold (float): Threshold factor for edge detection.
    #         plot (bool): If True, displays the image with the fitted ellipse overlay.
    #         savepath (str): If provided, the plot is saved to this path.

    #     Returns:
    #         (center_x, center_y, a, b, theta, pupil_mask)
    #         where (center_x, center_y) is the ellipse center,
    #                 a and b are the semimajor and semiminor axes,
    #                 theta is the rotation angle in radians,
    #                 pupil_mask is a 2D boolean array (True = inside ellipse).
    #     """
    #     # Normalize the image
    #     if image.max() != 0:

    #         image = image / image.max()

    #     image[ image > np.quantile(image, 0.98) ] = np.median( image )

    #     # Smooth the image
    #     smoothed_image = gaussian_filter(image, sigma=sigma)

    #     # Compute gradients (Sobel-like edge detection)
    #     grad_x = np.gradient(smoothed_image, axis=1)
    #     grad_y = np.gradient(smoothed_image, axis=0)
    #     edges = np.sqrt(grad_x**2 + grad_y**2)

    #     # Threshold edges to create a binary mask
    #     binary_edges = edges > (threshold * edges.max())

    #     # Get edge pixel coordinates
    #     y_coords, x_coords = np.nonzero(binary_edges)

    #     # Initial guess: center from mean, radius from average distance, and theta = 0.
    #     def initial_guess(x, y):
    #         center_x = np.mean(x)
    #         center_y = np.mean(y)
    #         r_init = np.sqrt(np.mean((x - center_x)**2 + (y - center_y)**2))
    #         return center_x, center_y, r_init, r_init, 0.0  # (cx, cy, a, b, theta)

    #     # Ellipse model function with rotation.
    #     def ellipse_model(params, x, y):
    #         cx, cy, a, b, theta = params
    #         cos_t = np.cos(theta)
    #         sin_t = np.sin(theta)
    #         x_shift = x - cx
    #         y_shift = y - cy
    #         xp =  cos_t * x_shift + sin_t * y_shift
    #         yp = -sin_t * x_shift + cos_t * y_shift
    #         # Model: xp^2/a^2 + yp^2/b^2 = 1 => residual = sqrt(...) - 1
    #         return np.sqrt((xp/a)**2 + (yp/b)**2) - 1.0

    #     # Fit via least squares.
    #     guess = initial_guess(x_coords, y_coords)
    #     result, _ = leastsq(ellipse_model, guess, args=(x_coords, y_coords))
    #     center_x, center_y, a, b, theta = result

    #     # Create a boolean pupil mask for the fitted ellipse
    #     yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
    #     cos_t = np.cos(theta)
    #     sin_t = np.sin(theta)
    #     x_shift = xx - center_x
    #     y_shift = yy - center_y
    #     xp = cos_t * x_shift + sin_t * y_shift
    #     yp = -sin_t * x_shift + cos_t * y_shift
    #     pupil_mask = (xp/a)**2 + (yp/b)**2 <= 1

    #     if savepath is not None:
    #         # Overlay for visualization
    #         overlay = np.zeros_like(image)
    #         overlay[pupil_mask] = 1

    #         plt.figure(figsize=(6, 6))
    #         plt.imshow(image, cmap="gray", origin="upper")
    #         plt.contour(binary_edges, colors="cyan", linewidths=1)
    #         plt.contour(overlay, colors="red", linewidths=1)
    #         plt.scatter(center_x, center_y, color="blue", marker="+")
    #         plt.title("Detected Pupil with Fitted Ellipse")
    #         #if savepath is not None:
    #         plt.savefig(savepath)
    #         #plt.show()
    #         plt.close()
    #     return center_x, center_y, a, b, theta, pupil_mask

    def open_all_shutters(self):
        msg = f"h_shut open 1,2,3,4"
        self._send_and_get_response(msg)

    def autoalign_coarse_parallel(self):
        # 1. shutter all beams off except 1, pause for a short time
        # 2. find the centre of the blob in the image
        # 3. save the pixel offsets
        # 4. repeat for all beams
        pixel_offsets = {}

        msg = f"h_shut close 2"
        self._send_and_get_response(msg)
        msg = f"h_shut close 3"
        self._send_and_get_response(msg)
        msg = f"h_shut close 4"
        self._send_and_get_response(msg)
        msg = f"h_shut open 1"
        self._send_and_get_response(msg)
        time.sleep(self._shutter_pause_time)

        flux_radius = 6

        frames = []

        for target_beam in range(1, 5):
            print(f"doing beam {target_beam}")
            if target_beam > 1:
                msg = f"h_shut close {target_beam-1}"
                self._send_and_get_response(msg)
                msg = f"h_shut open {target_beam}"
                self._send_and_get_response(msg)
                time.sleep(self._shutter_pause_time)

            blob_centre, flux, frame = self._get_blob(
                radius=flux_radius, return_flux=True, inc_frame=True
            )

            frames.append(frame)
            # to try
            # center_x, center_y, _, _, _,_ = detect_pupil(flux_radius, sigma=2, threshold=0.5, plot=False, savepath=None)

            if flux < self.flux_threshold:
                print(
                    f"Beam {target_beam} has low flux: {flux}. setting delta to zero."
                )
                pixel_offsets[target_beam] = np.array([0, 0])
            else:
                print(f"Beam {target_beam} flux: {flux:.2f}, centre: {blob_centre}")
                # calculate the pixel offsets from the target pixels
                pixel_offsets[target_beam] = np.array(
                    [
                        self.target_pixels[1] - blob_centre[0],
                        self.target_pixels[0] - blob_centre[1],
                    ]
                )

        if self.savepth is not None:
            np.savez(os.path.join(self.savepth, "heim_autoalign.npz"), frames=frames)

        # 5. unshutter all beams
        self.open_all_shutters()

        if self.output == "mcs":
            print("Sending offset commands to MCS...")
            self._send_offsets_to_mcs(pixel_offsets)
        elif self.output == "internal":
            time.sleep(self._shutter_pause_time)
            # 6. move them using the offsets + moveimage like calculation
            # key here is to parallelise
            uv_commands = {}
            for beam, offset in pixel_offsets.items():
                uv_cmd = asgE.move_img_calc("c_red_one_focus", beam, offset)
                uv_commands[beam] = uv_cmd

            # send commands
            axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]
            axes = [
                [axis + str(beam_number) for axis in axis_list]
                for beam_number in range(1, 5)
            ]

            for beam, uv_cmd in uv_commands.items():
                cmd = f"moverel {axes[beam-1][0]} {uv_cmd[0]}"
                self._send_and_get_response(cmd)
                cmd = f"moverel {axes[beam-1][2]} {uv_cmd[2]}"
                self._send_and_get_response(cmd)

            time.sleep(0.4)

            for beam, uv_cmd in uv_commands.items():
                cmd = f"moverel {axes[beam-1][1]} {uv_cmd[1]}"
                self._send_and_get_response(cmd)
                cmd = f"moverel {axes[beam-1][3]} {uv_cmd[3]}"
                self._send_and_get_response(cmd)

    # helper methods for pupil alignment fitting
    @staticmethod
    def fit_func(x, m, a, b, c):
        return -m * np.abs(x - a) - m * np.abs(x - b) + c

    @staticmethod
    def get_peak_flux_loc(pos, fluxes):
        fluxes /= np.max(fluxes)  # normalise
        params, _ = curve_fit(HeimdallrAA.fit_func, pos, fluxes, p0=[1, -0.05, 0.05, 1])

        return (params[1] + params[2]) / 2

    def imalign_single(self, beam_n):
        blob_centre = self._get_blob()

        pix_offset = np.array(
            [
                self.target_pixels[1] - blob_centre[0],
                self.target_pixels[0] - blob_centre[1],
            ]
        )
        uv_cmd = asgE.move_img_calc("c_red_one_focus", beam_n, pix_offset)

        axis_list = ["HTPP", "HTTP", "HTPI", "HTTI"]
        axes = [axis + str(beam_n) for axis in axis_list]

        cmd = f"moverel {axes[0]} {uv_cmd[0]}"
        self._send_and_get_response(cmd)
        cmd = f"moverel {axes[2]} {uv_cmd[2]}"
        self._send_and_get_response(cmd)
        time.sleep(0.5)
        cmd = f"moverel {axes[1]} {uv_cmd[1]}"
        self._send_and_get_response(cmd)
        cmd = f"moverel {axes[3]} {uv_cmd[3]}"
        self._send_and_get_response(cmd)
        time.sleep(0.5)

    def autoalign_pupil(self, beam):

        mv_time = 2.5

        unused_beams = [b for b in range(1, 5) if b != beam]

        msg = f"h_shut close {','.join(map(str, unused_beams))}"
        self._send_and_get_response(msg)
        msg = f"h_shut open {beam}"
        self._send_and_get_response(msg)
        time.sleep(self._shutter_pause_time)

        # 2. rough align beam
        self.imalign_single(beam)
        time.sleep(2)

        # 3. move pupil to optimize flux
        pup_offset = 0.2  # mm
        n_samp = 9
        flux_beam_radius = 6  # pixels

        measurement_locs_x = np.linspace(-pup_offset, pup_offset, n_samp)
        relative_measurement_locs = np.array(
            [
                measurement_locs_x[i] - measurement_locs_x[i - 1]
                for i in range(1, n_samp)
            ]
        )
        relative_measurement_locs = np.concatenate(
            [[-pup_offset], relative_measurement_locs]
        )

        fluxes_x = []
        for delta in relative_measurement_locs:
            cmd = f"mv_pup c_red_one_focus {beam} {delta} {0.0}"
            self._send_and_get_response(cmd)
            print("sent", cmd)
            time.sleep(mv_time)
            centre, flux = self._get_blob(radius=flux_beam_radius, return_flux=True)

            # if we are far from centre pixel, imalign the beam again
            if np.linalg.norm(centre - np.array(self.target_pixels)[::-1]) > 10:
                print(
                    f"Beam {beam} centroid {centre} far from target {self.target_pixels[::-1]}. Re-aligning image."
                )
                self.imalign_single(beam)
                time.sleep(2)
                centre, flux = self._get_blob(radius=flux_beam_radius, return_flux=True)

            fluxes_x.append(flux)

            # check if centre is close to the edge of the frame
            for bound in self.col_bnds:
                if np.abs(centre[1] - bound) < 10:
                    print(
                        f"Warning: Beam {beam} centroid close to edge of frame ({centre[1]} pixels). Alignment may be unreliable."
                    )

        cur_pupil_pos = measurement_locs_x[-1]

        fluxes_x = np.array(fluxes_x)
        if np.max(fluxes_x) < self.flux_threshold:
            print(
                f"Beam {beam} has low flux: {np.max(fluxes_x)}. Setting pupil offset back to original."
            )
            del_needed = -cur_pupil_pos
            cmd = f"mv_pup c_red_one_focus {beam} {del_needed} {0.0}"
            self._send_and_get_response(cmd)
            return

        optimal_offset_x = self.get_peak_flux_loc(measurement_locs_x, fluxes_x)

        del_needed = optimal_offset_x - cur_pupil_pos

        print(
            f"Optimal pupil offset for beam {beam} x: {optimal_offset_x:.4f} mm ({del_needed:.4f} mm from current position)"
        )
        cmd = f"mv_pup c_red_one_focus {beam} {del_needed} {0.0}"
        self._send_and_get_response(cmd)

        time.sleep(2)

        fluxes_y = []
        for delta in relative_measurement_locs:
            cmd = f"mv_pup c_red_one_focus {beam} {0.0} {delta}"
            self._send_and_get_response(cmd)
            print("sent", cmd)
            time.sleep(2.5)
            centre, flux = self._get_blob(radius=flux_beam_radius, return_flux=True)

            if np.linalg.norm(centre - np.array(self.target_pixels)[::-1]) > 10:
                print(
                    f"Beam {beam} centroid {centre} far from target {self.target_pixels[::-1]}. Re-aligning image."
                )
                self.imalign_single(beam)
                time.sleep(2)
                centre, flux = self._get_blob(radius=flux_beam_radius, return_flux=True)

            fluxes_y.append(flux)

            for bound in self.col_bnds:
                if np.abs(centre[1] - bound) < 10:
                    print(
                        f"Warning: Beam {beam} centroid close to edge of frame ({centre[1]} pixels). Alignment may be unreliable."
                    )

        fluxes_y = np.array(fluxes_y)

        optimal_offset_y = self.get_peak_flux_loc(measurement_locs_x, fluxes_y)
        cur_pupil_pos = measurement_locs_x[-1]

        del_needed = optimal_offset_y - cur_pupil_pos

        print(
            f"Optimal pupil offset for beam {beam} y: {optimal_offset_y:.4f} mm ({del_needed:.4f} mm from current position)"
        )
        cmd = f"mv_pup c_red_one_focus {beam} {0.0} {del_needed}"
        self._send_and_get_response(cmd)

        # now do x axis again to refine
        time.sleep(2)

        fluxes_x2 = []
        for delta in relative_measurement_locs:
            cmd = f"mv_pup c_red_one_focus {beam} {delta} {0.0}"
            self._send_and_get_response(cmd)
            print("sent", cmd)
            time.sleep(mv_time)
            centre, flux = self._get_blob(radius=flux_beam_radius, return_flux=True)

            if np.linalg.norm(centre - np.array(self.target_pixels)[::-1]) > 10:
                print(
                    f"Beam {beam} centroid {centre} far from target {self.target_pixels[::-1]}. Re-aligning image."
                )
                self.imalign_single(beam)
                time.sleep(2)
                centre, flux = self._get_blob(
                    radius=flux_beam_radius, return_flux=True
                )

            fluxes_x2.append(flux)

            for bound in self.col_bnds:
                if np.abs(centre[1] - bound) < 10:
                    print(
                        f"Warning: Beam {beam} centroid close to edge of frame ({centre[1]} pixels). Alignment may be unreliable."
                    )

        fluxes_x2 = np.array(fluxes_x2)

        optimal_offset_x2 = self.get_peak_flux_loc(measurement_locs_x, fluxes_x2)
        cur_pupil_pos = measurement_locs_x[-1]
        del_needed = optimal_offset_x2 - cur_pupil_pos
        print(
            f"Refined optimal pupil offset for beam {beam} x: {optimal_offset_x2:.4f} mm ({del_needed:.4f} mm from current position)"
        )

        cmd = f"mv_pup c_red_one_focus {beam} {del_needed} {0.0}"
        self._send_and_get_response(cmd)
        time.sleep(2)

        # saving
        if self.savepth is not None:
            np.savez(
                os.path.join(self.savepth, f"heimdallr_pupil_beam{beam}.npz"),
                meas_locs_x=measurement_locs_x,
                meas_locs_y=measurement_locs_x,
                fluxes_x=fluxes_x,
                fluxes_y=fluxes_y,
                optimal_offset_x=optimal_offset_x,
                optimal_offset_y=optimal_offset_y,
                measurement_locs_x2=measurement_locs_x,
                fluxes_x2=fluxes_x2,
                optimal_offset_x2=optimal_offset_x2,
            )

        return {
            "meas_locs_x": measurement_locs_x,
            "meas_locs_y": measurement_locs_x,
            "fluxes_x": fluxes_x,
            "fluxes_y": fluxes_y,
            "optimal_offset_x": optimal_offset_x,
            "optimal_offset_y": optimal_offset_y,
            "measurement_locs_x2": measurement_locs_x,
            "fluxes_x2": fluxes_x2,
            "optimal_offset_x2": optimal_offset_x2,
        }

    def autoalign_pupil_all(self, plot):
        # just like autoalign_3_pupil but for all beams
        datas = {}
        for beam in range(1, 5):
            datas[beam] = self.autoalign_pupil(beam)
            # open all shutters
            self.open_all_shutters()

        if plot:
            self.plot_autoalign_pupil_all(datas)

    @staticmethod
    def plot_autoalign_pupil_all(data_all):

        n_beams = 4
        fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

        for beam in range(1, n_beams + 1):
            data = data_all[beam]
            positions = [data[f"meas_locs_{i}"] for i in ["x", "y", "x"]]
            fluxes = [data[f"fluxes_{i}"] for i in ["x", "y", "x2"]]
            optimal_offsets = [data[f"optimal_offset_{i}"] for i in ["x", "y", "x2"]]
            ax = axs.flat[beam - 1]
            ax.plot(positions[0], fluxes[0], "o-", label="x", color="C0")
            ax.plot(positions[1], fluxes[1], "o-", label="y", color="C1")
            ax.plot(positions[2], fluxes[2], "o-", label="x2", color="C2")
            for i in range(3):
                ax.axvline(optimal_offsets[i], color=f"C{i}", ls="--")
            ax.set_title(f"Beam {beam}")
            if beam in [3, 4]:
                ax.set_xlabel("Position")
            if beam in [1, 3]:
                ax.set_ylabel("Flux")
            if beam == 1:
                ax.legend()
            ax.grid()

        plt.show()

    def _send_offsets_to_mcs(self, pixel_offsets):
        """
        convert pixel offsets to image offsets (pixels to arcsec)
        dummy conversion matrix for now

        Parameters
        ----------
        pixel_offsets : dict
            dictionary with beam number as key and pixel offset as value (np.array of shape (2,))
        """

        # dummy conversion matrix TODO: per beam matrix 2x2
        pix_to_arcsec = 2.40
        arcsec_offsets = {
            beam: offset * pix_to_arcsec for beam, offset in pixel_offsets.items()
        }

        # if any value is bigger than 100 in magnitude, set it to zero and print a warning
        for beam, offset in arcsec_offsets.items():
            offset_length = np.linalg.norm(offset)
            if offset_length > 100:
                print(
                    f"Warning: Beam {beam} offset too large ({offset_length:.2f} arcsec). Setting to zero."
                )
                arcsec_offsets[beam] = np.array([0.0, 0.0])

        # these are the hdlr_x_offset, hdlr_y_offset - need to reformat from dict
        # of beams to two lists - one for x offsets, one for y offsets
        # note that offset are flipped! x on the detector is y on sky
        x_offsets = [-arcsec_offsets[beam][1] for beam in range(1, 5)]
        y_offsets = [arcsec_offsets[beam][0] for beam in range(1, 5)]

        msg = {
            "origin": "s_h-autoalign",
            "data": [
                {"hdlr_x_offset": x_offsets},
                {"hdlr_y_offset": y_offsets},
                {"hdlr_complete": 1},
            ],
        }

        self.send_and_recv_ack(msg)

    def send_and_recv_ack(self, msg):
        # recieve ack
        print(f"sending {msg}")
        resp = self.mcs_client.send_payload(msg, decode_ascii=False)
        if resp is None or resp.get("ok") == False:
            print(resp)
            print("Failed to send offsets to MCS")
        else:
            print("msg acked")

    def test_mcs(self):
        # set "hdlr_x_offset" to 4 random numbers
        x_offsets = np.random.uniform(-1, 1, size=4).tolist()

        msg = {
            "origin": "s_h-autoalign",
            "data": [
                {"hdlr_x_offset": x_offsets},
                {"hdlr_complete": 1},
            ],
        }

        self.send_and_recv_ack(msg)


def main():
    parser = argparse.ArgumentParser(description="Autoalign Heimdallr beams.")
    parser.add_argument(
        "--shutter_pause_time",
        type=float,
        default=2.5,
        help="Seconds to pause after shuttering (default: 2.5)",
    )
    parser.add_argument(
        "-a",
        "--align",
        type=str,
        required=True,
        choices=[
            "cp",
            "coarseparallel",
            "ia",
            "imageall",
            "p3",
            "p1",
            "pupil3",
            "pa",
            "pupilall",
            "test_mcs",
        ],
        help="Alignment method: 'ia'/'imageall', 'pa'/'pupilall' or 'p3'/'pupil3'",
    )
    parser.add_argument(
        "-b",
        "--band",
        type=str,
        default="K2",
        choices=["K1", "K2"],
        help="Band to use: 'K1' or 'K2' (default: K2)",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        default="None",
        help="Path to save the alignment results (default: current directory)",
    )

    parser.add_argument(
        "-p",
        "--plot",
        type=bool,
        default=False,
        help="if results should be plotted to the screen when done (only valid for pa)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="internal",
        choices=["internal", "mcs", "none"],
        help="If the actuation should be done internally, or offset commands sent to MCS (default: internal)",
    )

    parser.add_argument(
        "-n",
        "--ncubes",
        type=int,
        default=3,
        help="Number of cubes to use, each of 200 frames (default: 3)",
    )

    parser.add_argument(
        "-t",
        "--t-pause",
        type=float,
        default=0.8,
        help="Seconds to pause between cubes when averaging (default: 0.8)",
    )

    args = parser.parse_args()

    heimdallr_aa = HeimdallrAA(
        shutter_pause_time=args.shutter_pause_time,
        band=args.band,
        flux_threshold=200.0,
        savepth=args.save_path,
        output=args.output,
        ncubes=args.ncubes,
        t_pause=args.t_pause,
    )

    if args.output == "internal":
        init_vals = heimdallr_aa.get_init_motors_state()

    try:
        if args.align in ["cp", "coarseparallel"]:
            print("New command name, rerun with -a ia or -a imageall instead")
        elif args.align in ["ia", "imageall"]:
            heimdallr_aa.autoalign_coarse_parallel()
        elif args.align in ["p3", "pupil3"]:
            if args.output == "mcs":
                print("Pupil alignment with MCS output not supported, exiting...")
                return
            heimdallr_aa.autoalign_pupil(3)
            # open all shutters
            heimdallr_aa.open_all_shutters()
        elif args.align in ["p1", "pupil1"]:
            if args.output == "mcs":
                print("Pupil alignment with MCS output not supported, exiting...")
                return
            heimdallr_aa.autoalign_pupil(1)
            # open all shutters
            heimdallr_aa.open_all_shutters()
        elif args.align in ["test_mcs"]:
            heimdallr_aa.test_mcs()
        elif args.align in ["pa", "pupilall"]:
            if args.output == "mcs":
                print("Pupil alignment with MCS output not supported, exiting...")
                return
            heimdallr_aa.autoalign_pupil_all(plot=args.plot)
        else:
            raise ValueError("Unknown alignment method.")

        print("Autoalignment completed.")
    except Exception as e:
        print(f"Error during alignment: {e}")
        # open all shutters
        heimdallr_aa.open_all_shutters()

        if args.output == "internal":
            print("Restoring initial motor positions.")
            heimdallr_aa.set_complete_state(init_vals)
            print("Initial motor positions restored.")


if __name__ == "__main__":
    main()
