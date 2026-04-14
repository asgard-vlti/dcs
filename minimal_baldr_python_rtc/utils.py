import numpy as np
import hcipy
import scipy.optimize as opt
import minimal_baldr_python_rtc.consts as consts
import zmq


def smooth_circle(grid, radius, softening=0.1, centre=(0, 0)):
    r = np.sqrt((grid.x - centre[0]) ** 2 + (grid.y - centre[1]) ** 2)
    return 1 / (1 + np.exp((r - radius) / softening))


def xcor_sum_model(params, args):
    img, grid, softening = args
    img /= np.sum(img)
    model = smooth_circle(
        grid, radius=params[0], softening=softening, centre=(params[1], params[2])
    ).reshape(grid.shape)
    model /= model.sum()
    return -np.sum(img * model)


def fit_pupil_centre(pupil_img):
    cam_grid = hcipy.make_pupil_grid(32, diameter=32)

    res = opt.minimize(
        xcor_sum_model,
        x0=[8, 0, 0],
        args=((pupil_img.reshape(consts.img_shape), cam_grid, 0.5),),
        bounds=((8, 8), (-10, 10), (-10, 10)),
    )
    img_center = np.array([15.5, 15.5])
    pupil_center = np.array([res.x[1], res.x[2]]) + img_center

    return pupil_center, img_center, cam_grid


def mds_connect(host: str, port: int = 5555, timeout_ms: int = 5000):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.connect(f"tcp://{host}:{port}")
    return ctx, sock


def mds_send(sock, msg: str) -> str:
    sock.send_string(msg)
    return sock.recv_string().strip()
