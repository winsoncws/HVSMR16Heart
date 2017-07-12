# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:37:23 2017

@author: winsoncws

Attempt 3D ConvNet
"""
from __future__ import division, print_function, absolute_import

from tensorgraph.layers import Conv3D, Conv2D, RELU, MaxPooling, LRN, Tanh, Dropout, \
                               Softmax, Flatten, Linear, TFBatchNormalization, Sigmoid
#from tensorgraph.utils import same
#from tensorgraph.node import StartNode, HiddenNode, EndNode
#from tensorgraph.graph import Graph
#from tensorgraph.layers.merge import Concat, Mean, Sum, NoChange
#import tensorgraph as tg
#from tensorgraph import Sequential
import tensorflow as tf
#from tensorgraph.cost import entropy, accuracy, iou, smooth_iou
#from math import ceil
from conv3D import Conv3D_Tranpose1, MaxPool3D, SoftMaxMultiDim, Residual3D, \
InceptionResnet_3D, ResidualBlock3D

from tensorflow.python.ops import gen_nn_ops


def filter(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1), name='filter')  
    
def bias(size):
    return tf.Variable(tf.zeros([size]), name='_b')                                
   
def getShape(input):
    shape = ()
    for i in input.shape:
        shape += (i.value,)
    return shape[1:4]
    
def maxPool3D(input, ksize, stride, padding='SAME'):
    output = tf.nn.max_pool3d(input,(1,)+ksize+(1,),(1,)+stride+(1,),'SAME')    
    return output, getShape(output)

def conv3d(input, channels, filters, ksize, stride, padding='SAME'):
    # input : b,d,h,w,in
    # filter_shape : d,h,w,in,out
    filter_shape = ksize + (channels, filters)    
    filter = tf.Variable(tf.random_normal(filter_shape, stddev=0.1), name='filter')
    bias = tf.Variable(tf.zeros([filters]), name='bias')  
    output = tf.nn.conv3d(input, filter, strides=(1,)+stride+(1,), padding=padding)
    return tf.nn.bias_add(output, bias)
    
def conv3d_Tr(input, channels, filters, output, ksize, stride, padding='SAME'):
    # input : b,d,h,w,in
    # filter_shape : d,h,w,out,in
    # output : b,d,h,w,out
    filter_shape = ksize + (filters, channels)
    batch_size = tf.shape(input)[0]
    d,h,w = output
    output_shape = tf.stack((batch_size,d,h,w,filters))
    print('stacked')   
    filter = tf.Variable(tf.random_normal(filter_shape, stddev=0.1), name='filter')
    bias = tf.Variable(tf.zeros([filters]), name='bias')  
    output = tf.nn.conv3d_transpose(input, filter, output_shape, strides=(1,)+stride+(1,), padding=padding)
    print('tranpose')    
    return tf.nn.bias_add(output, bias)


#phase = tf.placeholder(tf.bool, name='phase')

def batchNorm(input, training=True):
    out = tf.contrib.layers.batch_norm(input, decay=0.9, epsilon=1e-5,
                                        scale=True, is_training=training, scope=None)
    return tf.nn.relu(out, 'ReLU')  

def dense_batch_relu(x, phase, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, 100, 
                                               activation_fn=None,
                                               scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1, 
                                          center=True, scale=True, 
                                          is_training=phase,
                                          scope='bn')
        return tf.nn.relu(h2, 'relu')               
               
def batch_relu(x, phase, scope):
#    tf.nn.fused_batch_norm
    with tf.variable_scope(scope):
        out = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase, scope=None)
        return tf.nn.relu(out, 'ReLU')     


def model(input, train=True):
    with tf.name_scope('WMH'):
        convStride = (1,1,1)
        kSize3 = (3,3,3)
        poolStride = (2,2,2)
        poolSize = (2,2,2)
        inputChannel = 1
        #input = tf.placeholder('float32', [None, 20, 20, 20, 1])
        shape1 = getShape(input)
        conv1= conv3d(input, channels=inputChannel, filters=8, ksize=kSize3, stride=convStride)
        print('----------')
        print(conv1.shape)
        conv1, shape2 = maxPool3D(conv1,poolSize,poolStride,'SAME')
        print(shape2)
        conv1 = batchNorm(conv1, training=train) # with RELU
        #conv1 = batch_relu(conv1, phase=train,'BN')
        print(conv1.shape)
        conv1 = conv3d(conv1, channels=conv1.shape[4].value, filters=16, ksize=kSize3, stride=convStride)
        print(conv1.shape)
        conv1 = batchNorm(conv1, training=train)  # with RELU
        conv1 = conv3d_Tr(conv1, channels=conv1.shape[4].value, filters=8, output=shape1, ksize=kSize3, stride=poolStride)
        print(conv1.shape)
        conv1 = batchNorm(conv1, training=train)  # with RELU
        conv1 = conv3d(conv1, channels=conv1.shape[4].value, filters=3, ksize=kSize3, stride=convStride)
        print(conv1.shape)
        conv1 = tf.nn.softmax(conv1)
    return conv1


class BBB():
    def __init__(self, input):
        self.input = input
    
class AAA(BBB):
    def __init__(self, data):
        self.data = data
        self.add = self.addition(4)
        self.bbb = BBB(self.data)
    
    def update(self, data):
        self.data = data
        
    def addition(self,input):
        return input + self.data
    
    property
    def output(self):
        return self.bbb.input
