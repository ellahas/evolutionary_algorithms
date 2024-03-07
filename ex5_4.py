import numpy as np

tsp_file = "file-tsp.txt"
file_reader = open(tsp_file, "r")


def read_coords(file_reader):
    """Read the coordinates of the cities given in the file."""
    lines = file_reader.readlines()
    coords = np.zeros((len(lines), 2))
    for i, line in enumerate(lines):
        coords[i,:] = [float(c) for c in line.split()]
    return coords


def distance(coords1, coords2):
    """Calculate the eulerian distance between two coordinate pairs."""
    return np.sqrt((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2)


def fitness(path, coords):
    """Determine the fitness of a path based on the total distance travelled."""
    total_distance = 0
    for i in range(len(path)-1):
        total_distance += distance(coords[path[i], :], coords[path[i+1], :])
    return 1 / total_distance


coords = read_coords(file_reader)

print(fitness(list(range(len(coords))), coords))
