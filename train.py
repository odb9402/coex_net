import tensorflow as tf
import os
import sys
import numpy as np
import networkx as nx
import logging
import progressbar as pgb
from utils.util_funcs import *
from models.define_model import *
from models.define_graph import Coex_graph

def main():
    model_name = "test"
    LOG_DIR = os.path.join(os.getcwd(), 'tensorLog')
    logger.info("TF LOG will be saved at %s." % LOG_DIR)
    
    if os.path.exists(LOG_DIR) == False:
        os.mkdir(LOG_DIR)
    else:
        shutil.rmtree(LOG_DIR, ignore_errors=True)
        os.mkdir(LOG_DIR)
    ### Make graph with coexpression data
    graph = load_humanbase_coex_data("data/tooth_top_0.15")

    ### Give score features that represent relations with a disease for each vertex.
    #graph = update_disease_features_coex("data/Periodontitis_genes.csv", graph)
    graph = update_disease_features_coex("data/diabetes_genes.csv", graph)
    #graph = update_disease_features_coex("data/rheumatoid_genes.csv", graph)
    graph = update_gene_features_coex("data/all_gene_disease_associations.tsv", graph)
    
    coex_graph = Coex_graph(graph, "perio_graph", vertex_feature_num=3)
    
    train_nodes, test_nodes = split_train_test(coex_graph, test_ratio=0.2)
    logger.info("TEST # : {}, TRAIN # {}".format(len(test_nodes), len(train_nodes)))
    
    pn_rate = pos_neg_ratio(coex_graph)
    logger.info("PN RATE :: {}".format(pn_rate))
    
    #### Tensorflow Session and training.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.summary.scalar("Loss",loss) 
    tf.summary.scalar("Test_loss", eval_loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOG_DIR, graph=sess.graph)
    
    loss_mean = 0.0
    test_loss_mean = 0.0
    coex_graph.feed_data_load(coex_graph.random_node(),(FLAGS.n1_features, FLAGS.n2_features))
    coex_graph.vis_sampled_nodes()
    exit()
    #### Training.
    for i in range(FLAGS.epochs):
        train_dict = mini_batch_load(train_nodes, coex_graph, batch_size=FLAGS.batch_size)
        
        summary, _ = sess.run([merged, train_step], feed_dict=train_dict)
        train_loss, train_preds = sess.run([loss, prediction], feed_dict=train_dict)
        loss_mean += train_loss
        
        test_dict = mini_batch_load(test_nodes, coex_graph, batch_size=FLAGS.batch_size, is_test=True)            
        test_loss, test_preds = sess.run([eval_loss, test_prediction], feed_dict=test_dict)
        test_loss_mean += test_loss
        ### Test.
        if (i+1) % FLAGS.eval_every == 0:
            logger.info('Generation # {}. Train loss: {:.4f} , Test loss : {:.4f}'
                        .format(i+1, float(loss_mean/FLAGS.eval_every), float(test_loss/FLAGS.eval_every)))
            test_examples = list(test_dict.values())
            pos_samples = 0
            for j in range(FLAGS.batch_size):
                if float(test_examples[0][j]) > 0.0:
                    print(test_preds[j], test_examples[0][j])
                    pos_samples += 1
            j = 0
            while pos_samples > 0 and j < FLAGS.batch_size:
                if float(test_examples[0][j]) == 0.0:
                    print(test_preds[j], test_examples[0][j])
                    pos_samples -= 1
                j += 1
            loss_mean = 0.0
            test_loss_mean = 0.0
        writer.add_summary(summary, i)
    
    test_points = []
    
    bar = pgb.ProgressBar(max_value=100)
    for j in range(100):
        test_dict = mini_batch_load(test_nodes, coex_graph, batch_size=FLAGS.batch_size, is_test=True)            
        test_loss, test_preds = sess.run([eval_loss, test_prediction], feed_dict=test_dict)
        test_examples = list(test_dict.values())
        pos_samples = 0
        for k in range(FLAGS.batch_size):
            if float(test_examples[0][k]) > 0.0:
                test_points.append((test_preds[k], test_examples[0][k]))
                pos_samples += 1
        k = 0
        while pos_samples > 0 and k < FLAGS.batch_size:
            if float(test_examples[0][k]) == 0.0:
                test_points.append((test_preds[k], test_examples[0][k]))
                pos_samples -= 1
            k += 1
        bar.update(j)
    
    output_file = open("samples.txt",'w')
    for p in test_points:
        output_file.write("{}\t{}\n".format(p[0][0][0], p[1][0][0]))
    output_file.close()
    
    saver = tf.train.Saver()
    save_path = saver.save(sess, os.getcwd() + "/saved_models/{}.ckpt".format(model_name))
    logger.info("Model is saved in path : %s" % save_path)
    
if __name__ == '__main__':
    logger = logging.getLogger("COEX_GRAPH_LOG")
    logger.setLevel(logging.DEBUG)               # The logger object only output logs which have
                                                 # upper level than INFO.
    log_format = logging.Formatter('%(asctime)s:%(message)s')

    stream_handler = logging.StreamHandler()    # Log output setting for the command line.
    stream_handler.setFormatter(log_format)     # The format of stream log will follow this format.
    logger.addHandler(stream_handler)

    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("USER INTERRUPT. \n")
        sys.exit()