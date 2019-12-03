import os
import numpy as np
import tensorflow as tf
import tflearn

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
    return RecordWriter(data_loader, StandardProcessTup, num_of_threads=os.cpu_count() - 1)

class Model(object):
    def __init__(self, batch_size, reuse=False, tf_record_dir=None, num_epochs=0, weighting=[1]*2):
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
        # Using TFLearn wrappers for network building
        init = tflearn.initializations.xavier()
        with tf.variable_scope("level1", reuse=reuse):
            net1 = tflearn.layers.conv_3d(X, 8, 3, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='conv1-1-bmode', weights_init=init)
            net1 = tflearn.activation(net1, 'prelu')

        with tf.variable_scope("level2", reuse=reuse):
            net2_in = tflearn.layers.conv_3d(net1, 16, 2, strides=2, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='ds1', weights_init=init)
            net2 = tflearn.layers.conv_3d(net2_in, 16, 3, activation='prelu',padding='same', regularizer='L2',reuse=reuse,scope='conv2-1', weights_init=init)
            net2 = tflearn.layers.conv_3d(net2, 16, 3, activation='prelu',padding='same', regularizer='L2',reuse=reuse,scope='conv2-2', weights_init=init)
            net2 = tflearn.activation(net2, 'prelu')
            net2 = tf.add(net2_in, net2)

        with tf.variable_scope("level3"):
            net3_in = tflearn.layers.conv_3d(net2, 32, 2, strides=2, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='ds2', weights_init=init)
            net3 = tflearn.layers.conv_3d(net3_in, 32, 3, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='conv3-1', weights_init=init)
            net3 = tflearn.layers.conv_3d(net3, 32, 3, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='conv3-2', weights_init=init)
            net3 = tflearn.layers.conv_3d(net3, 32, 3, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='conv3-3', weights_init=init)
            net3 = tflearn.activation(net3, 'prelu')
            net3 = tf.add(net3_in, net3)

        with tf.variable_scope("level4"):
            net4_in = tflearn.layers.conv_3d(net3, 64, 2, strides=2, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='ds3', weights_init=init)
            net4 = tflearn.layers.conv_3d(net4_in, 64, 3, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='conv4-1', weights_init=init)
            net4 = tflearn.layers.conv_3d(net4, 64, 3, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='conv4-2', weights_init=init)
            net4 = tflearn.layers.conv_3d(net4, 64, 3, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='conv4-3', weights_init=init)
            net4 = tflearn.activation(net4, 'prelu')
            net4 = tf.add(net4_in, net4)

        with tf.variable_scope("bottom"):
            netbot_in = tflearn.layers.conv_3d(net4, 128, 2, strides=2, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='ds4', weights_init=init)
            netbot = tflearn.layers.conv_3d(netbot_in, 128, 3, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='convbot-1', weights_init=init)
            netbot = tflearn.layers.conv_3d(netbot, 128, 3, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='convbot-2', weights_init=init)
            netbot = tflearn.layers.conv_3d(netbot, 128, 3, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='convbot-3', weights_init=init)
            netbot = tflearn.activation(netbot, 'prelu')
            netbot = tf.add(netbot_in, netbot)

        with tf.variable_scope("level4_up"):
            net4_up = tflearn.layers.conv.conv_3d_transpose(netbot, 64, 2, [net4.get_shape().as_list()[1]] * 3,
                                                            strides=2, activation='prelu', padding='same',
                                                            regularizer='L2', reuse=reuse, scope='trans1',
                                                            weights_init=init)
            net4_up_concat = tflearn.layers.merge_ops.merge([net4_up, net4], axis=4, mode='concat')
            net4_up = tflearn.layers.conv_3d(net4_up_concat, 128, 3, activation='prelu', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv5-1', weights_init=init)
            net4_up = tflearn.layers.conv_3d(net4_up, 128, 3, activation='prelu', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv5-2', weights_init=init)
            net4_up = tflearn.layers.conv_3d(net4_up, 128, 3, activation='prelu', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv5-3', weights_init=init)
            net4_up = tflearn.activation(net4_up, 'prelu')
            net4_up = tf.add(net4_up_concat, net4_up)

        with tf.variable_scope("level3_up"):
            net3_up = tflearn.layers.conv.conv_3d_transpose(net4_up, 32, 2, [net3.get_shape().as_list()[1]] * 3,
                                                            strides=2, activation='prelu', padding='same',
                                                            regularizer='L2', reuse=reuse, scope='trans2',
                                                            weights_init=init)
            net3_up_concat = tflearn.layers.merge_ops.merge([net3_up, net3], axis=4, mode='concat')
            net3_up = tflearn.layers.conv_3d(net3_up_concat, 64, 3, activation='prelu', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv6-1', weights_init=init)
            net3_up = tflearn.layers.conv_3d(net3_up, 64, 3, activation='prelu', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv6-2', weights_init=init)
            net3_up = tflearn.layers.conv_3d(net3_up, 64, 3, activation='prelu', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv6-3', weights_init=init)
            net3_up = tflearn.activation(net3_up, 'prelu')
            net3_up = tf.add(net3_up_concat, net3_up)

        with tf.variable_scope("level2_up"):
            net2_up = tflearn.layers.conv.conv_3d_transpose(net3_up, 16, 2, [net2.get_shape().as_list()[1]] * 3,
                                                            strides=2, activation='prelu', padding='same',
                                                            regularizer='L2', reuse=reuse, scope='trans3',
                                                            weights_init=init)
            net2_up_concat = tflearn.layers.merge_ops.merge([net2_up, net2], axis=4, mode='concat')
            net2_up = tflearn.layers.conv_3d(net2_up_concat, 32, 3, activation='prelu', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv7-1', weights_init=init)
            net2_up = tflearn.layers.conv_3d(net2_up, 32, 3, activation='prelu', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv7-2', weights_init=init)
            net2_up = tflearn.activation(net2_up, 'prelu')
            net2_up = tf.add(net2_up_concat, net2_up)

        with tf.variable_scope("level1_up"):
            net1_up = tflearn.layers.conv.conv_3d_transpose(net2_up, 8, 2, [net1.get_shape().as_list()[1]] * 3,
                                                            strides=2, activation='prelu', padding='same',
                                                            regularizer='L2', reuse=reuse, scope='trans4',
                                                            weights_init=init)
            net1_up_concat = tflearn.layers.merge_ops.merge([net1_up, net1], axis=4, mode='concat')
            net1_up = tflearn.layers.conv_3d(net1_up_concat, 16, 3, activation='prelu', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv8-1', weights_init=init)
            net1_up = tflearn.activation(net1_up, 'prelu')
            net1_up = tf.add(net1_up_concat, net1_up)

        with tf.variable_scope("out"):
            net_fc1 = tflearn.layers.conv_3d(net1_up, 8, 1, activation='prelu', padding='same', regularizer='L2',
                                             reuse=reuse, scope='fc1', weights_init=init)
            net_fc1 = tflearn.layers.normalization.batch_normalization(net_fc1, reuse=reuse, scope='batch_fc_1')
            net_fc1 = tflearn.activation(net_fc1, "prelu")
            net_out = tflearn.layers.conv_3d(net_fc1, 2, 1, activation='sigmoid', padding='same', regularizer='L2',
                                             reuse=reuse, scope='output', weights_init=init)

        return net_out, net1
