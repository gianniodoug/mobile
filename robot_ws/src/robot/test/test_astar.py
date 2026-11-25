# Tests for the A*
import math
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np
from robot.astar import astar

SQ2 = math.sqrt(2)


def path_cost(path):
    cost = 0.0
    for (y0, x0), (y1, x1) in zip(path[:-1], path[1:]):
        if y0 != y1 and x0 != x1:
            cost += SQ2
        else:
            cost += 1.0
    return cost


def check_path(blocked, path, start, goal):
    h, w = blocked.shape
    assert tuple(path[0]) == tuple(start)
    assert tuple(path[-1]) == tuple(goal)

    for y, x in path:
        assert 0 <= y < h and 0 <= x < w, f"{(y, x)} is outside the grid"
        assert not blocked[y, x], f"{(y, x)} is blocked"

    for (y0, x0), (y1, x1) in zip(path[:-1], path[1:]):
        dy, dx = y1 - y0, x1 - x0
        assert max(abs(dy), abs(dx)) == 1, f"bad step {(y0, x0)} -> {(y1, x1)}"

        # no corner cutting on the diagonals
        if dy != 0 and dx != 0:
            assert not blocked[y0, x1] and not blocked[y1, x0], f"corner cut {(y0, x0)} -> {(y1, x1)}"


def reference_cost(blocked, start, goal):
    # we use scipy's dijkstra on the same grid to know the shortest cost
    # inf if there is no path
    h, w = blocked.shape
    graph = lil_matrix((h * w, h * w))

    for y in range(h):
        for x in range(w):
            if blocked[y, x]:
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if dy == 0 and dx == 0:
                        continue
                    if not (0 <= ny < h and 0 <= nx < w) or blocked[ny, nx]:
                        continue
                    if dy != 0 and dx != 0 and (blocked[y, nx] or blocked[ny, x]):
                        continue
                    if dy != 0 and dx != 0:
                        graph[y * w + x, ny * w + nx] = SQ2
                    else:
                        graph[y * w + x, ny * w + nx] = 1.0

    dist = dijkstra(graph.tocsr(), indices=start[0] * w + start[1])
    return dist[goal[0] * w + goal[1]]


def test_start_is_goal():
    blocked = np.zeros((5, 5), dtype=bool)
    path = astar(blocked, (2, 2), (2, 2))
    assert [tuple(c) for c in path] == [(2, 2)]


def test_straight_line():
    blocked = np.zeros((10, 10), dtype=bool)
    path = astar(blocked, (0, 0), (0, 5))
    check_path(blocked, path, (0, 0), (0, 5))
    assert math.isclose(path_cost(path), 5.0)


def test_diagonal():
    blocked = np.zeros((10, 10), dtype=bool)
    path = astar(blocked, (0, 0), (5, 5))
    check_path(blocked, path, (0, 0), (5, 5))
    assert math.isclose(path_cost(path), 5 * SQ2)


def test_wall_with_gap():
    # wall at ix = 10, open only on the first row
    blocked = np.zeros((20, 20), dtype=bool)
    blocked[1:, 10] = True
    start, goal = (10, 2), (10, 17)

    path = astar(blocked, start, goal)
    check_path(blocked, path, start, goal)
    assert math.isclose(path_cost(path), reference_cost(blocked, start, goal))


def test_u_trap():
    # the U is open towards the robot and the goal is behind it
    blocked = np.zeros((30, 30), dtype=bool)
    blocked[8, 8:22] = True
    blocked[21, 8:22] = True
    blocked[8:22, 21] = True
    start, goal = (15, 3), (15, 26)

    path = astar(blocked, start, goal)
    check_path(blocked, path, start, goal)
    assert math.isclose(path_cost(path), reference_cost(blocked, start, goal))


def test_unreachable_goal():
    # goal inside a closed square
    blocked = np.zeros((15, 15), dtype=bool)
    blocked[5, 5:10] = True
    blocked[9, 5:10] = True
    blocked[5:10, 5] = True
    blocked[5:10, 9] = True
    assert astar(blocked, (0, 0), (7, 7)) is None


def test_blocked_start_or_goal():
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[0, 0] = True
    blocked[4, 4] = True
    assert astar(blocked, (0, 0), (2, 2)) is None
    assert astar(blocked, (2, 2), (4, 4)) is None


def test_no_corner_cutting():
    # the only way would be the diagonal between the two blocked cells
    blocked = np.array([[False, True],
                        [True, False]])
    assert astar(blocked, (0, 0), (1, 1)) is None


def test_grid_not_modified():
    blocked = np.zeros((10, 10), dtype=bool)
    blocked[3:7, 5] = True
    before = blocked.copy()
    astar(blocked, (5, 0), (5, 9))
    assert np.array_equal(blocked, before)


def test_random_grids():
    # 40 random maps with ~25% obstacles, the path has to be the shortest one
    rng = np.random.default_rng(0)

    for _ in range(40):
        blocked = rng.random((30, 30)) < 0.25
        free = np.argwhere(~blocked)
        start = tuple(free[rng.integers(len(free))])
        goal = tuple(free[rng.integers(len(free))])

        path = astar(blocked, start, goal)
        best = reference_cost(blocked, start, goal)

        if math.isinf(best):
            assert path is None
        else:
            check_path(blocked, path, start, goal)
            assert math.isclose(path_cost(path), best)
