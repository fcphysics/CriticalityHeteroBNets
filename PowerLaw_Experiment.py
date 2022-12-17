import numpy as np
#import cubewalkers as cw
from cana.boolean_network import BooleanNetwork, BooleanNode

from time import time

import csv
import sys

def power_law_struct(N=100, m=15, gamma=2.25):

    Pk = np.array([k**(-gamma) for k in range(1, m+1)])
    Pk /= sum(Pk)

    Kin = np.random.choice(range(1, m+1), p=Pk, size=N)

    inputs = np.array([np.random.choice(range(N), replace=False, size=k) for k in Kin], dtype=object)

    return inputs


def power_law_bn(N=100, m=15, gamma=2.25, bias=0.5):

    inputs = power_law_struct(N, m, gamma)

    logic = dict.fromkeys(range(N))

    for n in range(N):
        logic[n] = {'name': "x{}".format(n),
                    'in': list(inputs[n]),
                    'out': [int(np.random.rand() < bias) for _ in range(2**len(inputs[n]))]}

    return BooleanNetwork.from_dict(logic=logic)

def node_properties(node: BooleanNode):

    if node.k >= 2:
        ke = node.effective_connectivity(norm=False)
    else:
        ke = int(not node.constant)
    ks = sum(node.edge_effectiveness(bound="upper"))

    p = node.bias()
    var = p*(1.0-p)
    if var == 0.0:
        h = 0.0
    else:
        h = -p*np.log2(p) - (1.0-p)*np.log2(1.0-p)

    #return [node.k, ke, ke-s, len(node.outputs), np.sum(np.array(node.outputs).astype(int))]
    return [node.k, ke, ks, p, var, h]


## State Space Definition
gamma = np.arange(1.5, 2.5, 0.1)
bias = np.arange(0.05, 0.5, 0.05)

## Simulation Parameters
num_nets = 10
n_walkers = 1000
N = int(sys.argv[1])
expn = int(sys.argv[2])

## Output data - File name as number of nodes and experiment number
f = open('Results{n:d}/Results-{expn:d}.csv'.format(n=N, expn=expn), 'w')
writer = csv.writer(f)
## Include avgH, avgV
writer.writerow(['gamma', 'bias', 'Derrida', 'avgK', 'avgKe', 'avgS', 'avgP', 'avgH', 'avgV'])

## Placeholder Arrays
Dc = np.zeros(num_nets)
avgP = np.zeros(num_nets)
avgV = np.zeros(num_nets)
avgH = np.zeros(num_nets)
avgK = np.zeros((3, num_nets))

for g in gamma:
    for p in bias:
        for idx in range(num_nets):

            BN = power_law_bn(N=N, m=15, gamma=g, bias=p)

            Dc[idx] = BN.derrida_coefficient(nsamples=n_walkers)*N

            K = np.transpose([node_properties(node) for node in BN.nodes])
            for i in range(3):
                avgK[i][idx] = np.mean(K[i])

            avgP[idx] = np.mean(K[3])
            avgV[idx] = np.mean(K[4])
            avgH[idx] = np.mean(K[5])

        writer.writerow([g, p, Dc, avgK[0], avgK[1], avgK[2], avgP, avgH, avgV])

f.close()
