import os
import json
import math
import random
import glob
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_3d, conv_3d_transpose, max_pool_3d
from tflearn.layers.normalization import batch_normalization
from tflearn.activations import relu, sigmoid
from tflearn.layers.merge_ops import merge

from oxnnet.data_loader import StandardDataLoader
from oxnnet.record import RecordWriter, StandardProcessTup, RecordReader
from oxnnet.full_inferer import StandardFullInferer
from oxnnet.feats_writer import StandardFeatsWriter

train_eval_test_no = [75, 15, 10]
segment_size_in = np.array([64] * 3)
segment_size_out = segment_size_in
crop_by = 0
stride = np.array([32] * 3, dtype=np.int)

data_loader = StandardDataLoader(stride, segment_size_in)


def build_record_writer(data_dir, dir_type_flag):
    data_loader.read_data_dir(data_dir, train_eval_test_no)
    return RecordWriter(data_loader, StandardProcessTup, num_of_threads=os.cpu_count())


# Input block for first layer
def input_block(input_tensor, num_filters, reuse, scope):
    x_in = batch_normalization(input_tensor, reuse=reuse, scope=scope + "_batch0")
    x = relu(conv_3d(x_in, num_filters, 3, activation='linear', padding='same', reuse=reuse, scope=scope + "_1",
                     weights_init='xavier'))
    x = relu(batch_normalization(merge([x, x_in], 'elemwise_sum', name=scope + "_merge")))
    return x


# Output block for last layer
def output_block(input_tensor, num_filters, reuse, scope):
    x_in = relu(
        conv_3d(input_tensor, num_filters, 1, activation='linear', padding='same', reuse=reuse, scope=scope + "_1",
                weights_init='xavier'))
    x = conv_3d(x_in, 2, 1, activation='linear', padding='same', reuse=reuse, scope=scope + "_2", weights_init='xavier')
    return x


# Down convolutions with down sampling using strides
def down_block(input_tensor, num_filters, reuse, scope):
    # x_in = relu(batch_normalization(conv_3d(input_tensor, num_filters, 2, strides=2, activation='linear', padding='same', weights_init='xavier', scope=scope+"_1", reuse=reuse)))
    # x = relu(batch_normalization(conv_3d(x_in, num_filters, 3, activation='linear', padding='same', weights_init='xavier', scope=scope+"_2", reuse=reuse)))
    # x = relu(batch_normalization(merge([x, x_in], 'elemwise_sum', name=scope+"_merge")))
    x_in = relu(
        conv_3d(input_tensor, num_filters, 2, strides=2, activation='linear', padding='same', weights_init='xavier',
                scope=scope + "_1", reuse=reuse))
    x = relu(
        conv_3d(x_in, num_filters, 3, activation='linear', padding='same', weights_init='xavier', scope=scope + "_2",
                reuse=reuse))
    x = relu(merge([x, x_in], 'elemwise_sum', name=scope + "_merge"))
    return x


# De-convolutions with concatenation
def up_block(input_tensor, concat, num_filters, reuse, scope):
    x = conv_3d_transpose(input_tensor, num_filters, 2, strides=2, output_shape=[concat.get_shape().as_list()[1]] * 3,
                          activation='linear', padding='same', weights_init='xavier', scope=scope + "_1", reuse=reuse)
    x_in = tf.concat([x, concat], axis=4)
    x = conv_3d(x_in, num_filters * 2, 3, activation='linear', padding='same', reuse=reuse, scope=scope + "_2",
                weights_init='xavier')
    x = relu(merge([x, x_in], 'elemwise_sum', name=scope + "_merge"))
    # x = relu(batch_normalization(merge([x, x_in], 'elemwise_sum', name=scope+"_merge"), reuse=reuse, scope=scope+"batch")) - Giving OOM issues
    return x


class Model(object):
    def __init__(self, batch_size, reuse=False, tf_record_dir=None, num_epochs=0, weighting=[1] * 2):
        self.batch_size = batch_size
        record_reader = RecordReader(StandardProcessTup(data_loader))
        with tf.device('/cpu:0'):
            with tf.variable_scope("input"):
                if tf_record_dir:
                    x_shape = [-1] + list(segment_size_in) + [1]
                    y_shape = [-1] + list(segment_size_out) + [1]
                    if reuse:
                        X, Y = record_reader.input_pipeline(False, batch_size, None, tf_record_dir)
                    else:
                        X, Y = record_reader.input_pipeline(True, batch_size, num_epochs, tf_record_dir)
                    self.X = tf.reshape(X, x_shape)
                    self.Y = tf.reshape(Y, y_shape)
                else:
                    self.X = tf.placeholder(
                        dtype=tf.float32,
                        shape=[None] + segment_size_in.tolist() + [1])
                    self.Y = tf.placeholder(
                        dtype=tf.float32,
                        shape=[None] + segment_size_out.tolist() + [1])
                X = self.X
                Y = tf.cast(tf.one_hot(tf.reshape(tf.cast(self.Y, tf.uint8), [-1] + list(segment_size_out)), 2),
                            tf.float32)
                X_A = tf.split(X, 2)
                Y_A = tf.split(Y, 2)
        with tf.variable_scope("inference") as scope:
            losses = []
            preds = []
            for gpu_id in range(2):
                with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
                        if gpu_id == 0:
                            logits, self.feats = self.build_net(X_A[gpu_id], False)
                        else:
                            logits, self.feats = self.build_net(X_A[gpu_id], True)
                        with tf.variable_scope("pred"):
                            softmax_logits = tf.nn.softmax(logits)
                            pred = tf.cast(tf.argmax(softmax_logits, axis=4), tf.float32)
                            preds.append(pred)
                        with tf.variable_scope("dice"):
                            self.dice_op = tf.divide(2 * tf.reduce_sum(tf.multiply(softmax_logits, Y_A[gpu_id])),
                                                     tf.reduce_sum(pred) + tf.reduce_sum(Y_A[gpu_id]), name='dice')
                        with tf.variable_scope("loss"):
                            class_weight = tf.constant(weighting, tf.float32)
                            weighted_logits = tf.multiply(logits, tf.reshape(class_weight, [-1, 1, 1, 1, 2]))
                            loss_op = tf.nn.softmax_cross_entropy_with_logits(logits=weighted_logits,
                                                                              labels=Y_A[gpu_id])
                        # Choose the metrics to compute:
                        names_to_values, names_to_updates = tf.contrib.metrics.aggregate_metric_map({
                            'accuracy': tf.contrib.metrics.streaming_accuracy(softmax_logits, Y_A[gpu_id]),
                            'precision': tf.contrib.metrics.streaming_precision(softmax_logits, Y_A[gpu_id]),
                            'recall': tf.contrib.metrics.streaming_recall(softmax_logits, Y_A[gpu_id]),
                            'mse': tf.contrib.metrics.streaming_mean_squared_error(softmax_logits, Y_A[gpu_id])
                        })
                        losses.append(loss_op)
            self.loss_op = tf.reduce_mean(tf.concat(losses, axis=0))
            self.pred = tf.cast(tf.concat(preds, axis=0), tf.float32)
            self.mse = names_to_values['mse']
            with tf.variable_scope("metrics"):
                self.metric_update_ops = list(names_to_updates.values())
            if tf_record_dir:
                tf.summary.scalar('dice', self.dice_op)
                tf.summary.scalar('loss', self.loss_op)
            for metric_name, metric_value in names_to_values.items():
                op = tf.summary.scalar(metric_name, metric_value)

    def build_full_inferer(self):
        return StandardFullInferer(segment_size_in, segment_size_out, crop_by, stride, self.batch_size)

    def build_feats_writer(self):
        crop_by = 2
        stride_test = segment_size_in - crop_by
        segment_size_out = stride_test
        return StandardFeatsWriter(segment_size_in, segment_size_out, crop_by, stride_test, self.batch_size, 16)

    def build_net(self, X, reuse=False):
        # Encoder path
        with tf.variable_scope("level1") as scope:
            net = input_block(X, 16, reuse, "level1")
            lvl1 = net

        with tf.variable_scope("level2") as scope:
            net = down_block(net, 32, reuse, "level2")
            lvl2 = net

        with tf.variable_scope("level3") as scope:
            net = down_block(net, 64, reuse, "level3")
            lvl3 = net

        # Bottom level
        with tf.variable_scope("bottom") as scope:
            net = down_block(net, 128, reuse, "bottom")
            lvl4 = net

        # Decoder path
        with tf.variable_scope("level3_up") as scope:
            net = up_block(net, lvl3, 64, reuse, "level3_up")
        with tf.variable_scope("level2_up") as scope:
            net = up_block(net, lvl2, 32, reuse, "level2_up")
        with tf.variable_scope("level1_up") as scope:
            net = up_block(net, lvl1, 16, reuse, "level1_up")

        # Output level
        net = output_block(net, 16, reuse, "output_level")
        return net, lvl1
