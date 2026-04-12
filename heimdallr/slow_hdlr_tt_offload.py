"""
Modify tip/tilt. Algorithm...
1) Read in all the baseline averaged power for K1 and K2.
2) Add K1 and K2 together. Let's keep this simple!
3) Find the SNR of each image by noise = np.percentile(power,15),
and SNR = (np.max(power) - noise)/noise
4) Computer the centroids for baselines, and convert to
telescopes by weighted least squares.
5) Via an MDS connection, move the HTXI motors!
"""

import time
import zmq
import numpy as np

try:
    import ZMQ_control_client as Z
except:
    import heimdallr.ZMQ_control_client as Z

zmq_context = zmq.Context()

# MDS Socket
socket = zmq_context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 10000)
socket.connect("tcp://mimir:5555")
connected = True

xdevices = ["HTTI1", "HTTI2", "HTPI3", "HTTI4"]
xsigns = [1, 1, -1, 1]
ydevices = ["HTPI1", "HTPI2", "HTTI3", "HTPI4"]
ysigns = [-1, -1, -1, -1]
M_lacour = np.array(
    [
        [-1, 1, 0, 0],
        [-1, 0, 1, 0],
        [-1, 0, 0, 1],
        [0, -1, 1, 0],
        [0, -1, 0, 1],
        [0, 0, -1, 1],
    ]
)
M_tel_avg = np.abs(M_lacour) / 2


def step_tt(cmds, devices, signs):
    global connected
    if not connected:
        # Try to reconnect.
        try:
            socket.connect("tcp://mimir:5555")
            connected = True
        except:
            print("Failed to reconnect to MDS.")
            return
    for cmd, device, sign in zip(cmds, devices, signs):
        msg = f"tt_step {device} {int(round(-cmd * sign * 4))}"
        print(msg)
        try:
            socket.send_string(msg)
            print(socket.recv_string())  # acknowledgement
        except zmq.Again:
            print(f"Timeout while sending command: {msg}.")
            return
        time.sleep(0.01)


def get_baseline_powers():
    ims = np.zeros((6, 8, 8))
    for baseline in range(6):
        for filter in ["K1", "K2"]:
            ims[baseline] += Z.get_im(
                'get_baseline_im "{}", {}'.format(filter, baseline)
            )
    return ims


def compute_snr_and_centroid(im):
    noise = np.percentile(im, 15)
    snr = (np.max(im) - noise) / noise
    x = np.linspace(-4, 3, 8)
    xy = np.meshgrid(x, x)
    total_power = np.sum(im)
    x_centroid = np.sum(xy[0] * im) / total_power
    y_centroid = np.sum(xy[1] * im) / total_power
    return (x_centroid, y_centroid), snr


def telescope_centroids(baseline_centroids, snrs):
    # Convert baseline centroids to telescope centroids using least squares
    # one axis at a time.
    A = M_tel_avg * np.sqrt(snrs)[:, np.newaxis]  # Weight by sqrt(SNR)
    b = np.array(baseline_centroids).flatten() * np.sqrt(snrs)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    print(baseline_centroids)
    print(x)
    return x


# Just once - not in a loop for now
def main_loop():
    baseline_powers = get_baseline_powers()
    baseline_centroids = []
    snrs = []
    for i in range(6):
        centroid, snr = compute_snr_and_centroid(baseline_powers[i])
        baseline_centroids.append(centroid)
        snrs.append(snr)
    print(snrs, baseline_centroids)
    # Now we have the centroids and SNRs for each baseline, we can compute the telescope commands.
    # Do x then y separately.
    x_centroids = [c[0] for c in baseline_centroids]
    y_centroids = [c[1] for c in baseline_centroids]
    # First, x.
    telescope_cmds = telescope_centroids(x_centroids, snrs)
    step_tt(telescope_cmds, xdevices, xsigns)
    # Then, y.
    telescope_cmds = telescope_centroids(y_centroids, snrs)
    step_tt(telescope_cmds, ydevices, ysigns)


if __name__ == "__main__":
    main_loop()
