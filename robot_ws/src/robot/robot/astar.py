# The actual A* algo
import numpy as np
import math
import heapq


def astar(occ, start, goal):
    # special cases
    if occ[goal] or occ[start]:
        return None 
    if start == goal:
            return [start]
    frontier = [(0.0, start)]
    g = {start: 0.0}
    parent = {start: None}
    done = set()

    while frontier:
         
        _, examined = heapq.heappop(frontier)

        if examined in done:
            continue
        if examined == goal:
            return _recreate(parent, start, examined)
        done.add(examined)

        for neighbor in _neighbors(occ, examined):
            node, dist = neighbor
            if node not in g or g[node] > g[examined] + dist:
                g[node] = g[examined] + dist
                parent[node] = examined
                heapq.heappush(frontier, (g[node] + _heuristic(node, goal), node))

    return None

         


    

def _heuristic(cell, goal):
    y_cell, x_cell = cell
    y_goal, x_goal = goal

    return math.hypot(y_cell - y_goal, x_cell - x_goal)

def _neighbors(occ, cur_pos):
    neighbors = []
    iy, ix = cur_pos
    h, w = occ.shape

    for dy in [-1, 0, 1]:
         for dx in [-1, 0, 1]:
              if dx != 0 or dy != 0:
                   if not (0 <= iy + dy < h and 0 <= ix + dx < w):
                           continue
                   elif occ[(iy + dy, ix + dx)] == False:
                        if dx != 0 and dy != 0:
                             if occ[(iy, ix + dx)] or occ[(iy + dy, ix)]:
                                  continue
                             else:
                                  neighbors.append(((iy + dy, ix + dx), math.sqrt(2)))
                        else:
                             neighbors.append(((iy + dy, ix + dx), 1.0))

    return neighbors
                                  
                                  

def _recreate(parent, beginning, current):
    path_to_current = [current]
    my_node = current

    while my_node != beginning:
        path_to_current.append(parent[my_node])
        my_node = parent[my_node]

    path_to_current.reverse()
    
    return path_to_current
