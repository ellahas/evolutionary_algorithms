import numpy as np
import matplotlib.pyplot as plt

l = 100
nr_generations = 1500
mu = 1 / l

x = np.random.randint(2, size=l)

def invert_bits(x, mu):
    x_m = []
    for c in x:
        if np.random.rand() < mu:
            x_m.append(1-c)
        else:
            x_m.append(c)
    return x_m
    
#print(x)
#Task 5.2.1 - Algoritm 1 
def ga_1(x):
    results = []
    for g in range(nr_generations):
        x_m = invert_bits(x, mu)  
        if np.sum(x_m) > sum(x):  
            x = x_m
        results.append(np.sum(x))
    return results
    
##Task 5.2.2 - Algorithm 2 
def ga_2(x):
    results = []
    for g in range(nr_generations):
        x_m = invert_bits(x, mu)  
        x = x_m
        results.append(np.sum(x))
    return results
 
 
results_1 = ga_1(x)
results_2 = ga_2(x)


num_runs = 50
results1 = []
results2 = []

#5.2.4
for i in range(num_runs):
    results1.append(ga_1(x))
    results2.append(ga_2(x))


plt.figure(figsize=(10, 6))
plt.plot(np.mean(results1, axis=0), label="Algorithm1")
plt.plot(np.mean(results2, axis=0), label="Algorithm2")
plt.title("Average Performance Comparison over Multiple Runs")
plt.xlabel("Number of Generations")
plt.ylabel("Average Fitness")
plt.legend()
plt.grid(True)
plt.show()
