import math
import numpy as np
from robot.planner_utils import make_occ, world_to_cell, cell_to_world

OR_X, OR_Y, RES = -2.0, -1.0, 0.05
SHAPE = (100, 100)


def test_known_point():
    assert world_to_cell(1.37, 0.42, OR_X, OR_Y, RES, SHAPE) == (28, 67)

def test_outside_the_map():
    assert world_to_cell(77.77, -77.77, OR_X, OR_Y, RES, SHAPE) == None
    assert world_to_cell(-77.77, -77.77, OR_X, OR_Y, RES, SHAPE) == None
    assert world_to_cell(0, -77.77, OR_X, OR_Y, RES, SHAPE) == None
    assert world_to_cell(-77.77, 0, OR_X, OR_Y, RES, SHAPE) == None

def test_round_trip():
    x, y = cell_to_world(39, 77, OR_X, OR_Y, RES)
    assert world_to_cell(x, y, OR_X, OR_Y, RES, SHAPE) == (39, 77)

def test_inflation():

    grid = np.zeros((9, 9), dtype=np.int8)
    grid[4, 4] = 100
    occ = make_occ(grid, 65, 2)

    assert occ.sum() == 13
    assert occ[4, 6]
    assert not occ[4, 7]
    assert not occ[6, 6]
    assert grid[4, 4] == 100 and grid.sum() == 100
