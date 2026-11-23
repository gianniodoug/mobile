from scipy.ndimage import binary_dilation
import numpy as np
import math

def make_occ(grid, threshold, r):
    occ = (grid > threshold)

    ay, ax = np.mgrid[-r:r + 1, -r:r + 1]
    disk = ax**2 + ay**2 <= r**2

    return binary_dilation(occ, structure=disk)


def world_to_cell(x, y, origin_x, origin_y, resol, shape):

    ix = math.floor((x - origin_x) / resol)
    iy = math.floor((y - origin_y) / resol)

    h, w = shape
    if not (0 <= iy < h and 0 <= ix < w):
        return None
    return (iy, ix)


def cell_to_world(iy, ix, origin_x, origin_y, resol):
    # +0.5 is for us to end up in the centre of the cell
    # without it we would be going to the bottom left corner
    # and then there's a problem since were exactly on the edge of 2 cells
    # dont pay attention it's just a bug fix
    x = (ix + 0.5) * resol + origin_x
    y = (iy + 0.5) * resol + origin_y

    return (x, y)