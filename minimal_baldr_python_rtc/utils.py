import numpy as np


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
