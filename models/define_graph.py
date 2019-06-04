import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

class Coex_graph():
    def __init__(self, graph, name, vertex_feature_num=1, mode='train'):
        self.name = name
        self.graph = graph
        self.vertex_feature_num = vertex_feature_num
        self.mode = mode
        self.set_network_adj()
        self.set_network_features(target_feature='Periodontitis')

        self.A_hat = np.matmul(np.matmul(self.D_sqrt, self.vertex_labels),self.D_sqrt)
    
        
    def set_network_adj(self, adj=None, back_bone=True):
        if adj==None:
            self.adj = nx.convert_matrix.to_numpy_matrix(self.graph)
        
        self.D_sqrt = np.diag(np.sqrt(np.fromiter(dict(self.graph.degree).values(), float) + 1))
        back_bone = np.identity(self.adj.shape[0])
        self.adj = self.adj + back_bone
        
    def set_network_features(self, target_feature=None):
        if target_feature==None:
            self.vertex_labels = np.fromiter(dict(self.graph.nodes(data='score')).values(), float)
        else:
            self.vertex_labels = np.fromiter(dict(self.graph.nodes(data=target_feature)).values(), float)
        
    def remove_vertex_label_randomly(self, prob=0.8):
        labels = self.vertex_labels.copy()
        for i in range(len(labels)):     
            if random.random() < prob:
                 labels[i] = None
                    
        return labels
    
    def k_neighbors(self, node, k=1, sampling_num=None):
        """
        """
        i = 1
        try:
            neighbors = {i: set(random.sample(set(nx.classes.function.neighbors(self.graph, node)), sampling_num[i]))}
        except ValueError:
            print("The node {} has the less number of neighbors than {}.".format(node, sampling_num[i]))
            return None
        
        while i < k:
            neighbors[i+1] = set()
            for n in neighbors[i]:
                candidates = set(nx.classes.function.neighbors(self.graph, n))
                try:
                    sampled_nodes = set(random.sample(candidates, sampling_num[i+1]))
                except ValueError:
                    print("The node {} has the less number of neighbors than {}.".format(node, sampling_num[i+1]))
                    return None
                neighbors[i+1] = neighbors[i+1].union(sampled_nodes)
            for j in range(i):
                neighbors[i+1] = neighbors[i+1] - neighbors[j+1]
            i = i + 1
            
        return neighbors        
       
    def feed_data_load(self, node, node_nums=None):
        sampled_nodes = self.neighbor_sampling(node, node_nums[0])
        if sampled_nodes == None:
            return None
        
        depth1_nodes = list(sampled_nodes)
        self.node_list = [node] + depth1_nodes  
        
        depth2_nodes = []
        for i in depth1_nodes:
            sampled_nodes = self.neighbor_sampling(i,node_nums[1])
            if sampled_nodes == None:
                return None
            self.node_list += list(sampled_nodes)
            depth2_nodes.append(list(sampled_nodes))
            
        depth1_features = [ list(self.graph.nodes(data=True)[x].values()) for x in depth1_nodes ]
        depth2_features = []
        for i in depth2_nodes:
            depth2_features.append([list(self.graph.nodes(data=True)[x].values()) for x in i])
        
        depth1_edges = []
        for i in depth1_nodes:
            depth1_edges.append(self.graph[node][i]['weight'])
            
        depth2_edges = []
        for i, j in zip(depth1_nodes, depth2_nodes):
            edge_weights = []
            for k in j:
                edge_weights.append(self.graph[i][k]['weight'])
            depth2_edges.append(edge_weights)
        
        depth1_edges = np.array(depth1_edges)
        depth2_edges = np.array(depth2_edges)
        depth1_features = np.array(depth1_features)
        depth2_features = np.array(depth2_features)
        
        return depth1_features, depth1_edges, depth2_features, depth2_edges
       
    def vis_sampled_nodes(self):
        node_list = self.node_list
        k = self.graph.subgraph(node_list)
        pos = nx.spring_layout(self.graph)
        
        nx.draw_networkx_nodes(k, pos, node_list=k.nodes())
        
        plt.axis('off')
        plt.savefig('visual_sample')
        
    
    def neighbor_sampling(self, node, sampling_num=None):
        candidates = set(nx.classes.function.neighbors(self.graph, node))
        try:
            sampled_nodes = set(random.sample(candidates, sampling_num))
        except ValueError:
            #print("The node {} has the less number of neighbors than {}.".format(node, sampling_num))
            return None
        
        return sampled_nodes

    def random_node(self):
        return random.choice(list(self.graph.nodes()))