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

    ke = node.effective_connectivity(norm=False)
    s = sum(node.edge_effectiveness(bound="upper"))

    return [node.k, ke, ke-s]


## State Space Definition
gamma = np.arange(1.5, 2.5, 0.1)
bias = np.arange(0.05, 0.5, 0.05)

## Simulation Parameters
num_nets = 10
n_walkers = 1000

## Output data
f = open('Results-{}.csv'.format(sys.argv[1]), 'w')
writer = csv.writer(f)
writer.writerow(['gamma', 'bias', 'Derrida', 'avgK', 'medK', 'avgKe', 'medKe', 'avgKc', 'medKc'])

## Placeholder Arrays
Dc = np.zeros(num_nets)
avgK = np.zeros((3, num_nets))
medK = np.zeros((3, num_nets))

for g in gamma:
    for p in bias:
        for idx in range(num_nets):

            BN = power_law_bn(N=100, m=15, gamma=g, bias=p)
            #luts, inputs = cw.conversions.cana2cupyLUT(BN)

            #cw_model = cw.Model(lookup_tables=luts, node_regulators=inputs, n_time_steps=1, n_walkers=n_walkers)

            #Dc[idx] = cw_model.derrida_coefficient(threads_per_block=(16, 16))
            start = time()
            Dc[idx] = BN.derrida_coefficient(nsamples=n_walkers)*100
            print("Time to Derrida: {}".format(time()-start))

            K = np.transpose([node_properties(node) for node in BN.nodes])
            for i in range(3):
                avgK[i][idx] = np.mean(K[i])
                medK[i][idx] = np.median(K[i]) 

            print("Time to measure: {}".format(time()-start))

        writer.writerow([g, p, Dc, avgK[0], medK[0], avgK[1], medK[1], avgK[2], medK[2]])

f.close()
