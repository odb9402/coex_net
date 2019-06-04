import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import pandas as pd
import progressbar as pgb
import csv
import random
import os, sys
import shutil
from utils.util_funcs import *
from models.define_graph import Coex_graph

### Hyperparameters of the model.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0004, '')
flags.DEFINE_integer('output_dim', 1, '')
flags.DEFINE_integer('feature_num', 3, '')
flags.DEFINE_integer('batch_size', 100, '')
flags.DEFINE_integer('max_steps', 1, '')
flags.DEFINE_float('dropout', 0.6, '')
flags.DEFINE_integer('eval_every', 20, '')
flags.DEFINE_integer('epochs', 3000, '')
flags.DEFINE_integer('neighbors_1', 20, '')
flags.DEFINE_integer('neighbors_2', 15, '')

flags.DEFINE_integer('n2_features', 30,'')
flags.DEFINE_integer('n1_features', 40,'')

### Tensor shape :: (batch_size, number of node, feature_num)

target_node = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 1, FLAGS.output_dim), name="Target_node_features")

neighbor1 = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.neighbors_1, FLAGS.feature_num), name="Neighbor1_node_features")
edge_weight1 = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.neighbors_1), name="Edge_weights_neighbor1")

neighbor2 = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.neighbors_1, FLAGS.neighbors_2, FLAGS.feature_num), name="Neighbor2_node_features")
edge_weight2 = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.neighbors_1, FLAGS.neighbors_2), name="Edge_weights_neighbor2")
is_train_step = tf.placeholder(tf.bool)
p_dropout = tf.placeholder(tf.float32)

loss_weight= tf.placeholder(tf.float32)

neighbor2_weight = tf.get_variable("neighbor_weight_21", shape=[FLAGS.feature_num, FLAGS.n2_features], 
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform='True'))
neighbor2_weight2 = tf.get_variable("neighbor_weight_22", shape=[FLAGS.n2_features, FLAGS.n2_features], 
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform='True'))
neighbor1_weight = tf.get_variable("neighbor_weight_11", shape=[FLAGS.feature_num, FLAGS.n1_features], 
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform='True'))
neighbor1_weight2 = tf.get_variable("neighbor_weight_12", shape=[FLAGS.n1_features, FLAGS.n1_features], 
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform='True'))
aggregated2_weight1 = tf.get_variable("aggregated21_weight", shape=[FLAGS.n1_features+FLAGS.n2_features, 10],
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform='True'))
aggregated2_weight2 = tf.get_variable("aggregated22_weight", shape=[10, FLAGS.output_dim], 
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform='True'))

neighbor2_bias = tf.get_variable("neighbor_bias_21", shape=[FLAGS.n2_features],
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform='True'))
neighbor2_bias2 = tf.get_variable("neighbor_bias_22", shape=[FLAGS.n2_features],
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform='True'))
neighbor1_bias = tf.get_variable("neighbor_bias_11", shape=[FLAGS.n1_features],
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform='True'))
neighbor1_bias2 = tf.get_variable("neighbor_bias_12", shape=[FLAGS.n1_features],
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform='True'))
aggregated2_bias = tf.get_variable("aggregated2_bias1", shape=[10],
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform='True'))
aggregated2_bias2 = tf.get_variable("aggregated2_bias2", shape=[FLAGS.output_dim],
                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform='True'))

#target_weight = tf.get_variable("target_weight", shape=[FLAGS.feature_num, FLAGS.feature_num],
#                    initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform='True'))

def tensorSummaries(name, tensor):
    with tf.name_scope(name):
        mean = tf.reduce_mean(tensor)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(tensor))
        tf.summary.scalar('min', tf.reduce_min(tensor))
        tf.summary.histogram('histogram', tensor)

tensorSummaries("Neighbor1_weight1", neighbor1_weight)
tensorSummaries("Neighbor1_weight2",neighbor1_weight2)
tensorSummaries("Neighbor2_weight1",neighbor2_weight)
tensorSummaries("Neighbor2_weight2",neighbor2_weight2)
tensorSummaries("AGG_weight1",aggregated2_weight1)
tensorSummaries("AGG_weight2",aggregated2_weight2)
tensorSummaries("Neighbor1_bias1",neighbor1_bias)
tensorSummaries("Neighbor1_bias2",neighbor1_bias2)
tensorSummaries("Neighbor2_bias1",neighbor2_bias)
tensorSummaries("Neighbor2_bias2",neighbor2_bias2)
tensorSummaries("AGG_bias1",aggregated2_bias)
tensorSummaries("AGG_bias2",aggregated2_bias2)


def aggregator2(neighbors1, neighbors2, edge_weights, neighbor_weights, neighbor_bias, node_weights, node_bias):
    edge_weights_expanded = tf.expand_dims(edge_weights, -1)
    weighted_neighbors = tf.multiply(neighbors2, edge_weights_expanded)
    neighbors2_mean = tf.reduce_mean(weighted_neighbors, axis=2)
    
    neighbors2_mean = tf.reshape(neighbors2_mean, [FLAGS.batch_size * FLAGS.neighbors_1, FLAGS.feature_num])
    neighbors2_mean = tf.matmul(neighbors2_mean, neighbor_weights)
    neighbors2_mean = tf.nn.bias_add(neighbors2_mean, neighbor_bias)
    
    #neighbors2_mean = tf.nn.leaky_relu(neighbors2_mean)
    #neighbors2_mean = tf.matmul(neighbors2_mean, neighbor2_weight2)
    #neighbors2_mean = tf.nn.bias_add(neighbors2_mean, neighbor2_bias2)
    
    neighbors1 = tf.reshape(neighbors1, [FLAGS.batch_size * FLAGS.neighbors_1, FLAGS.feature_num])
    neighbors1 = tf.matmul(neighbors1, node_weights)
    neighbors1 = tf.nn.bias_add(neighbors1, node_bias)
    
    #neighbors1 = tf.nn.leaky_relu(neighbors1)
    #neighbors1 = tf.matmul(neighbors1, neighbor1_weight2)
    #neighbors1 = tf.nn.bias_add(neighbors1, neighbor1_bias2)
    
    neighbors2_mean = tf.reshape(neighbors2_mean, [FLAGS.batch_size, FLAGS.neighbors_1, FLAGS.n2_features])
    neighbors1 = tf.reshape(neighbors1, [FLAGS.batch_size, FLAGS.neighbors_1, FLAGS.n1_features])
    
    concated_node = tf.concat([neighbors2_mean, neighbors1], axis=2)
    concated_node = tf.nn.leaky_relu(concated_node)
    concated_node = tf.nn.dropout(concated_node, p_dropout)
    
    return concated_node
    
    
def aggregator1(neighbors, edge_weights, neighbor_weights, neighbor_bias):
    edge_weights_expanded = tf.expand_dims(edge_weights, -1)
    weighted_neighbors = tf.multiply(neighbors, edge_weights_expanded)
    neighbors_mean = tf.reduce_mean(weighted_neighbors, axis=1)
    
    neighbors_mean = tf.matmul(neighbors_mean, neighbor_weights)
    neighbors_mean = tf.nn.bias_add(neighbors_mean, aggregated2_bias)
    neighbors_mean = tf.nn.leaky_relu(neighbors_mean)
    #neighbors_mean = tf.nn.sigmoid(neighbors_mean)
    neighbors_mean = tf.nn.dropout(neighbors_mean, p_dropout)
    
    neighbors_mean = tf.matmul(neighbors_mean, aggregated2_weight2)
    
    target_node_prediction = tf.reshape(neighbors_mean, [FLAGS.batch_size, 1, FLAGS.output_dim])
    target_node_prediction = tf.nn.bias_add(target_node_prediction, neighbor_bias)
    #target_node_prediction = tf.nn.relu(neighbors_mean)
    
    return target_node_prediction


def gcn_model():
    first_aggregated_nodes = aggregator2(neighbor1, neighbor2, edge_weight2,
                                         neighbor2_weight, neighbor2_bias,
                                         neighbor1_weight, neighbor1_bias)
    target_node = aggregator1(first_aggregated_nodes, edge_weight1, aggregated2_weight1, aggregated2_bias2)
    return target_node
    
def model_summary():
    model_vars = tf.trainable_variables()
    tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    
model_output = gcn_model()
test_model_output = gcn_model()

prediction = tf.nn.sigmoid(model_output)
test_prediction = tf.nn.sigmoid(test_model_output)

loss = tf.reduce_mean(
    tf.nn.weighted_cross_entropy_with_logits(targets=target_node, logits=model_output, pos_weight=loss_weight))

eval_loss = tf.reduce_mean(
    tf.nn.weighted_cross_entropy_with_logits(targets=target_node, logits=test_model_output, pos_weight=loss_weight))

#loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target_node, logits=model_output))
#eval_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target_node, logits=model_output))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train_step = optimizer.minimize(loss)

model_summary()
print(FLAGS)    