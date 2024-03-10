import random
import string
import numpy as np
import matplotlib.pyplot as plt

random.seed(10)

L = 15  
N = 200  
K = 2  
pc = 1  
G_max = 100  
alphabets = string.ascii_lowercase + string.ascii_uppercase
mu_values = [0, 1/L, 3/L]  
nr_runs = 10  
target_string = "heLloIamaTarget"


def create_population(str_length=15, size=200):
    population = []
    while len(population) < size:
        individual = ''
        for i in range(str_length):
            individual += random.choice(alphabets)
        population.append(individual)
    return population
    

def calculate_fitness(individual):
    correct_characters = 0
    for a, b in zip(individual, target_string):
        if a == b:
            correct_characters += 1
    fitness = correct_characters / L
        
    return fitness


def select_parents(population, repeat=50):
    selected_parents = []
    for i in range(N):
        participants = random.sample(population, K)
        fitness = []
        for p in participants:
            fit = calculate_fitness(p)
            fitness.append(fit)
            
        best_fit_idx = np.argmax(fitness)
        fittest_ind = participants[best_fit_idx]
        selected_parents.append(fittest_ind)
    
    return selected_parents


def mutate(child, mu):
    new_child = ''
    for char in child:
        rand = random.uniform(0, 1)
        if rand < mu:
            new_child += random.choice(alphabets)
        else:
            new_child += char
    return new_child


def calc_diversity(population):
    distances = []
    for i, ind_1 in enumerate(population):
        for j, ind_2 in enumerate(population):
            if i != j:
                distance = 0
                for a, b in zip(ind_1, ind_2):
                    if a != b:
                        distance += 1
                distances.append(distance)
    return np.mean(distances)


def run_algorithm(mu):
    t_finish = []
    gen_diversity= []

    for _ in range(nr_runs):
        population = create_population()
        generation = 0

        while generation < G_max:
            fitness = [calculate_fitness(each) for each in population]
            if max(fitness) == 1:
                t_finish.append(generation)
                print('Target found for: ', ' mu:', mu, 'at generation=', generation)
                break
                
            selected_parents = select_parents(population, fitness)
            new_population = []

            for i in range(0, N, 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1]
                rand = random.random()
                if rand < pc:
                    rand_number = random.randint(1, len(parent1) - 1)
                    child1 = parent1[:rand_number] + parent2[rand_number:]
                    child2 = parent2[:rand_number] + parent1[rand_number:]
                    child1 = mutate(child1, mu)
                    child2 = mutate(child2, mu)
                    new_population.extend([child1, child2])
                else:
                    new_population.extend([parent1, parent2])

            population = new_population
            if generation % 10 == 0:
                gen_diversity.append(calc_diversity(population))
            generation += 1

    return t_finish, gen_diversity


results = []
for mu in mu_values:
    _,res = run_algorithm(mu)
    results.append(res)

plt.figure(figsize=(8, 6))
for i in range(len(results)):
    res = results[i]
    mu = mu_values[i]
    plt.plot(res, label=f'μ = {mu}')

plt.title('Population Diversity Over Generations')
plt.xlabel('Generation')
plt.ylabel('Mean Pairwise Hamming Distance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()