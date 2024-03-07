import numpy as np

tsp_file = "file-tsp.txt"
file_reader = open(tsp_file, "r")


def read_coords(file_reader):
    """Read the coordinates of the cities given in the file."""
    lines = file_reader.readlines()
    coords = np.zeros((len(lines), 2))
    for i, line in enumerate(lines):
        coords[i, :] = [float(c) for c in line.split()]
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


def order_crossover(p1, p2):
    """Generate two children from two parents with order crossover."""
    # choose two cut points
    cuts = np.random.choice(range(1, len(p1)), size=2, replace=False)
    cuts = np.sort(cuts)

    # keep middle piece
    c1 = np.zeros(len(p1), int) - 1
    c1[cuts[0]:cuts[1]] = p1[cuts[0]:cuts[1]]
    c2 = np.zeros(len(p2), int) - 1
    c2[cuts[0]:cuts[1]] = p2[cuts[0]:cuts[1]]

    # take complement of other parent
    complement_p2 = np.setdiff1d(p2, c1, assume_unique=True)
    complement_p1 = np.setdiff1d(p1, c2, assume_unique=True)

    # fill gaps in order, from second cut
    for i in range(len(c1)):
        j = (i + cuts[1]) % len(c1)
        if c1[j] == -1:
            c1[j] = complement_p2[i]

    for i in range(len(c2)):
        j = (i + cuts[1]) % len(c2)
        if c2[j] == -1:
            c2[j] = complement_p1[i]
    return c1, c2


def mutation(child):
    """Mutate a child by swapping two random positions."""
    positions = np.random.choice(len(child), size=2)  # allows the same position twice, so there is some chance of no mutation as well
    mutated_child = child.copy()
    mutated_child[positions[0]] = child[positions[1]]
    mutated_child[positions[1]] = child[positions[0]]
    return mutated_child


coords = read_coords(file_reader)
