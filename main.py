from src.generate_data import generate_data
from src.subcase import structure, effect
import numpy as np

# configuration of data
n_samples = 50000
distribution = 'laplace' # 'laplace' or 'beta'
graph = 'a' # 'a' or 'b' or 'c' or 'd' or 'e' or 'f' or 'g' or 'h'


latent, observed = 1, 3
print(distribution, 'Fig. 3(', graph, ')')
data, weights, w_id = generate_data(graph, n_samples=n_samples, distribution=distribution, seed=2026)
Z, T, O = data[:, 0], data[:, 1], data[:, 2]
graph_pred = structure(T, O, Z)
if 'f' in graph_pred:
    print('unidentifiable')
else:
    weight_pred = effect(T, O, Z, graph_pred)
    weight_true = weights[w_id]
    print('true weight:', weight_true, 'predicted weight:', weight_pred)

