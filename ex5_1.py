import numpy as np
import matplotlib.pyplot as plt


def reproduction_probs(f, xs):
    results = np.zeros(len(xs))
    for i in range(len(xs)):
        results[i] = f(xs[i])
    probs = results / np.sum(results)
    return probs, results


def plot_pies(functions, xs, function_names):
    fig = plt.figure()
    all_probs = np.zeros((len(xs), len(functions)))
    all_results = np.zeros((len(xs), len(functions)))
    for i in range(len(functions)):
        ax = plt.subplot(len(functions)//2, 2, i+1)
        probs, results = reproduction_probs(functions[i], xs)
        all_probs[:, i] = probs
        all_results[:, i] = results
        ax.pie(probs, labels=xs)
        ax.set_title(function_names[i])
    return fig, all_probs, all_results


functions = [lambda x: np.abs(x), lambda x: x**2, lambda x: 2*x**2, lambda x: x**2 + 20]
function_names = ["abs(x)", "x^2", "2x^2", "x^2 + 20"]
xs = [2, 3, 4]
fig, all_probs, all_results = plot_pies(functions, xs, function_names)
print(all_probs)
print(all_results)
plt.savefig("evolutionary_algorithms/pie_plots.png")
plt.show()
