import progressbar as pgb
import pandas as pd
import networkx as nx
import random
import math
from models.define_model import *

def load_humanbase_coex_data(file_name):
    """
    """
    target_file = open(file_name, 'r')
    target_raws = target_file.readlines()
    print("Reading coexpression data {}. . . .".format(file_name))
    target_list = list(map(lambda x : x.rstrip().split('\t') , target_raws))
    
    print("Edge building . . . . .")
    coex_graph = nx.Graph()
    bar = pgb.ProgressBar(max_value=len(target_raws))
    prog = 0
    for i in target_list:
        try:
            coex_graph.add_edge(i[0], i[1], weight=float(i[2]))
        except:
            print("{} is not valid float".format(i[2]))
        prog += 1
        bar.update(prog)
    
    return coex_graph


def update_disease_features_coex(file_name, coex_graph):
    """
    """
    genes = pd.read_csv(file_name)
    genes = genes.sort_values(by=['geneid'])
    disease_name = file_name.rsplit('/')[1].split('_')[0]
    
    print("Update features from {}. . .".format(file_name))
    bar = pgb.ProgressBar(max_value=len(coex_graph.nodes))
    prog = 0
    
    for v in coex_graph.nodes:
        ## Search matched geneid
        check_in = False
        start = 0
        end = len(genes) - 1
        mid = int((start+end)/2)
        while (end-start) > 1:
            mid = int((start + end)/2)
            if int(v) > genes['geneid'].iloc[mid]:
                start = mid
            elif int(v) < genes['geneid'].iloc[mid]:
                end = mid
            else:
                nx.set_node_attributes(coex_graph, {v :{disease_name : float(genes['score'].iloc[mid])}})
                check_in = True
                break
        if not check_in:
            nx.set_node_attributes(coex_graph, {v :{disease_name : 0.0}})
            check_in = False
        prog += 1
        bar.update(prog)
    
    return coex_graph


def update_gene_features_coex(file_name, coex_graph):
    gene_table = pd.read_table(file_name)[['geneId', 'DSI', 'DPI']].drop_duplicates().dropna()
    
    print("Update DSI, DPI from {} . . .".format(file_name))
    bar = pgb.ProgressBar(max_value=len(coex_graph.nodes))
    prog = 0
    
    missed_nodes = []
    for v in coex_graph.nodes:
        check_in = False
        start = 0
        end = len(gene_table)
        mid = int((start+end)/2)
        while (end-start) > 1:
            mid = int((start+end)/2)
            if int(v) > gene_table['geneId'].iloc[mid]:
                start = mid
            elif int(v) < gene_table['geneId'].iloc[mid]:
                end = mid
            else:
                nx.set_node_attributes(coex_graph,
                                      {v: {'dpi' : float(gene_table['DPI'].iloc[mid]),
                                           'dsi': float(gene_table['DSI'].iloc[mid])}})
                check_in = True
                break
        if not check_in:
            #nx.set_node_attributes(coex_graph,{v: {'dpi' : 0.5, 'dsi': 0.5}})
            missed_nodes.append(v)
            check_in = False
            
        prog += 1
        bar.update(prog)
    for n in missed_nodes:
        coex_graph.remove_node(n)
        
    return coex_graph

def visualize_network(net, edge_width_scale=1):
    """
    
    """
    pos = nx.spring_layout(net)
    
    weight_list = []
    for node1, node2, data in net.edges(data=True):
        weight_list.append(data['weight'])
    weight_set = set(weight_list)

    for w in weight_set:
        weighted_edges = [(node1, node2) for (node1, node2, edge_attr) in net.edges(data=True) if edge_attr['weight']==w]
        width = edge_width_scale*w*len(net)/sum(weight_set)
        nx.draw_networkx_edges(net, pos, edge_list=weighted_edges, width=width)
    nx.draw_networkx_nodes(net, pos, node_list=net.nodes(), node_size=20)
    
    plt.axis('off')
    plt.show()

    
def pos_neg_ratio(graph):
    all_features = graph.graph.nodes(data=True)
    neg_nums = 0.0
    pos_nums = 0.0
    for n in all_features:
        for x in list(n[1].values())[0:1]:
            if float(x) == 0.0:
                neg_nums += 1
            else:
                pos_nums += 1
    return neg_nums/pos_nums
        

def split_train_test(graph, test_ratio=0.2):
    node_list = list(graph.graph.nodes())
    
    test_nodes = []
    train_nodes = []
    
    while len(node_list) > 0:
        if random.random() < test_ratio:
            test_nodes.append(node_list.pop())
        else:
            train_nodes.append(node_list.pop())
    
    return train_nodes, test_nodes


def mini_batch_load(candidate_nodes, graph, batch_size=1, is_test=False):
    target_node_batch = []
    neighbor1_batch = []
    edge_weight1_batch = []
    neighbor2_batch = []
    edge_weight2_batch = []
    
    for j in range(batch_size):
        while True:
            random_node = random.choice(candidate_nodes)
            random_node_features = np.array(
                list(graph.graph.nodes(data=True)[random_node].values()))[0].reshape(1,FLAGS.output_dim)
            feed_data = graph.feed_data_load(random_node, node_nums=(FLAGS.neighbors_1, FLAGS.neighbors_2))
            if feed_data != None:
                break
    
        target_node_batch.append(random_node_features)
        neighbor1_batch.append(feed_data[0])
        edge_weight1_batch.append(feed_data[1])
        neighbor2_batch.append(feed_data[2])
        edge_weight2_batch.append(feed_data[3])
        
    neighbor1_batch = np.array(neighbor1_batch)
    edge_weight1_batch = np.array(edge_weight1_batch)
    neighbor2_batch = np.array(neighbor2_batch)
    edge_weight2_batch = np.array(edge_weight2_batch)
    
    if is_test:
        dropout_prob = 1
    else:
        dropout_prob = 0.6 
        
    feed_forward_dict = {target_node: target_node_batch,
                  neighbor1: neighbor1_batch, #feed_data[0].reshape(neighbor1.shape),
                  edge_weight1: edge_weight1_batch, #feed_data[1].reshape(edge_weight1.shape),
                  neighbor2: neighbor2_batch, #feed_data[2].reshape(neighbor2.shape),
                  edge_weight2: edge_weight2_batch, #feed_data[3].reshape(edge_weight2.shape),
                  loss_weight: math.sqrt(pos_neg_ratio(graph)),
                  is_train_step: not is_test,
                  p_dropout: dropout_prob}
    
    return feed_forward_dict