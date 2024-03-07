import numpy as np

tsp_file = "evolutionary_algorithms/file-tsp.txt"
file_reader = open(tsp_file, "r")


def read_coords(file_reader):
    lines = file_reader.readlines()
    coords = np.zeros((len(lines), 2))
    for i, line in enumerate(lines):
        coords[i,:] = [float(c) for c in line.split()]
    return coords


def distance(coords1, coords2):
    """Calculate the eulerian distance between two coordinate pairs."""
    return np.sqrt((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2)


coords = read_coords(file_reader)
print(coords)
