import torch
import numpy as np
import networkx as nx
from src.utils import independence


def SelectPdf(Num,data_type):
    if data_type == "laplace":
        noise =np.random.laplace(0, 1, size=Num)
    elif data_type == "beta":
        noise = np.random.beta(0.33, 0.67, size=Num)
    else: #gauss
        noise = np.random.normal(0, 1, size=Num)
    return noise


def normalize(data):
    data -= np.mean(data)
    data /= np.std(data)
    return data


def generate_data(id, n_samples=1000, distribution='laplace', latent=1, observed=3, seed=0):
    if id == 'a':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        w_id = 3
    elif id == 'b':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        w_id = 4
    elif id == 'c':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
        w_id = 4
    elif id == 'd':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
        w_id = 3
    elif id == 'e':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
        w_id = 5
    elif id == 'f':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0]])
        w_id = 4
    elif id == 'g':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 0]])
        w_id = 5
    elif id == 'h':
        iv_adj = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 0]])
        w_id = 4
    else:
        raise ValueError
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    g = nx.DiGraph(iv_adj)
    n_weights = len(g.edges())
    while True:
        weights = torch.Tensor(n_weights).uniform_(-0.5, 0.5)
        for i in range(n_weights):
            if weights[i]>0:
                weights[i] += 0.5
            else:
                weights[i] -= 0.5
        
        adj = torch.zeros([latent+observed, latent+observed])
        for i in range(n_weights):
            adj[list(g.edges)[i][1], list(g.edges)[i][0]]=weights[i]
        mix = (torch.inverse(torch.eye(latent+observed) - adj))
        if np.all(np.abs(mix.numpy()[np.abs(mix.numpy()) > 1e-6]) > 0.2): # faithfulness
            break
    
    noises = []
    for i in range(latent+observed):
        while True:
            new_noise = normalize(SelectPdf(n_samples, distribution))
            if np.all(np.array([np.abs(np.corrcoef(new_noise, noise)[0, 1]) < 0.05 for noise in noises])) \
                and np.all(np.array([independence(new_noise, noise, 0.2)[0] for noise in noises])):
                noises.append(new_noise)
                break
    noises = np.stack(noises, axis=0)
    data = mix.matmul(torch.Tensor(noises)).t()
    data = data[:,range(latent, observed+latent)]
    
    return data.numpy(), weights.numpy(), w_id

