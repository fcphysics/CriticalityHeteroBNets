from cana.boolean_network import BooleanNetwork
from cana.datasets.bio import THALIANA, DROSOPHILA, BUDDING_YEAST, load_all_cell_collective_models
import pandas as pd
from time import time

node_cols = ["network", "node", "k", "ke", "keN", "s", "sN", "kc", "kcN", "bias"]
network_cols = ["network", "Nnodes", "k", "ke", "keN", "s", "sN", "kc", "kcN", "bias", "biasUnweighted", "dc"]

def weightedMeanBias(df):
    top = sum(df.apply(
        func = lambda s: s["bias"] * (2**s["k"]),
        axis=1
    ))
    bot = sum(df.apply(func=lambda s: 2**s["k"], axis=1))
    return top / bot

# loops through all networks and all nodes
# computes node level measures
# and some network level measures
def computeMeasures(networks):
    nodeData = pd.DataFrame(columns = node_cols)
    networkData = pd.DataFrame(columns = network_cols)
    numDudNodes = 0
    timeS = time()
    for i, network in enumerate(networks):
        print(f"on network {i+1} out of {len(networks)}, {network.name}, {len(network.nodes)} nodes")
        for node in network.nodes:
            if node.k == 0:
                numDudNodes += 1
                continue
            x = _computeMeasures(node, network)
            x["network"] = network.name
            # print(x)
            nodeData = pd.concat([nodeData, x])

        # compute network avgs and vars of node level measures here
        dc = network.derrida_coefficient(nsamples=1000) * network.Nnodes
        netMeans = nodeData[nodeData["network"]==network.name].drop(columns=["network", "node"]).mean()
        curNetNodes = nodeData[nodeData["network"]==network.name]
        x = pd.DataFrame({
            network_cols[0]: network.name,
            network_cols[1]: network.Nnodes,
            network_cols[2]: netMeans[network_cols[2]],
            network_cols[3]: netMeans[network_cols[3]],
            network_cols[4]: netMeans[network_cols[4]],
            network_cols[5]: netMeans[network_cols[5]],
            network_cols[6]: netMeans[network_cols[6]],
            network_cols[7]: netMeans[network_cols[7]],
            network_cols[8]: netMeans[network_cols[8]],
            network_cols[9]: weightedMeanBias(curNetNodes),
            network_cols[10]: netMeans["bias"],
            network_cols[11]: dc
        }, index=[0])
        networkData = pd.concat([networkData, x])

    print(f"there were {numDudNodes} nodes with k=0")
    print(time() - timeS)
    return nodeData, networkData

# computes node level measures of interest
# ke, S, kc, normalized and un-normalized
# bias, H(bias)
def _computeMeasures(node, network):
    ke = node.effective_connectivity(norm=False)
    s = node.sensitivity(norm=False)
    keNorm = node.effective_connectivity(norm=True)
    sNorm = node.sensitivity(norm=True)
    bias = node.bias()

    x = pd.DataFrame({
        # node_cols[0]: "placeholder",
        node_cols[1]: node.name,
        node_cols[2]: node.k,
        node_cols[3]: ke,
        node_cols[4]: keNorm,
        node_cols[5]: s,
        node_cols[6]: sNorm,
        node_cols[7]: ke - s,
        node_cols[8]: keNorm - sNorm,
        node_cols[9]: bias
    }, index=[0])

    return x

if __name__ == "__main__":
    # networks = [load_all_cell_collective_models()[1]]
    networks = load_all_cell_collective_models()
    nodeData, networkData = computeMeasures(networks)
    print(nodeData, networkData)
    nodeData.to_csv("cc_node_data.csv")
    networkData.to_csv("cc_network_data.csv")
    
