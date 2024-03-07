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


def tournament_selection(generation, coords, K):
    """Select two sets of K parents, best of each group procreates together."""
    parents = np.random.choice(generation.shape[0], K)
    best_p = None
    best_f = 0
    for p in parents:
        f = fitness(generation[p, :], coords)
        if f > best_f:
            best_f = f
            best_p = generation[p, :]
    return best_p


def generation_step(generation, coords, K=2):
    """Generate a new generation. Assumes generations of even number of individuals."""
    children = np.zeros(generation.shape, int)
    for i in range(generation.shape[0]//2):
        p1 = tournament_selection(generation, coords, K)
        p2 = tournament_selection(generation, coords, K)
        c1, c2 = order_crossover(p1, p2)
        c1 = mutation(c1)
        c2 = mutation(c2)
        children[i, :] = c1
        children[i+generation.shape[0]//2, :] = c2
    return children


def random_population(n_cities, N):
    """Generate a population of N random permutations of range(n_cities)"""
    generation = np.zeros((N, n_cities))
    for n in range(N):
        generation[n, :] = np.random.permutation(n_cities)
    return generation


coords = read_coords(file_reader)
start_generation = random_population(coords.shape[0], 50)
