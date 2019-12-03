import os

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_3d, conv_3d_transpose, max_pool_3d


from oxnnet.data_loader import StandardDataLoaderPowerDoppler
from oxnnet.record import RecordWriter, PowerDopplerProcessTup, RecordReader
from oxnnet.full_inferer import PowerDopplerFullInferer
from oxnnet.feats_writer import StandardFeatsWriter

train_eval_test_no = [75, 15, 10]
segment_size_in = np.array([64] * 3)
segment_size_out = segment_size_in
crop_by = 0
stride = np.array([32] * 3, dtype=np.int)

data_loader = StandardDataLoaderPowerDoppler(stride, segment_size_in)

def build_record_writer(data_dir, dir_type_flag):
    data_loader.read_data_dir(data_dir, train_eval_test_no)
    return RecordWriter(data_loader, PowerDopplerProcessTup, num_of_threads=os.cpu_count() - 1)


# Input block for first layer
def input_block(bmode, pd, num_filters, reuse, scope, init):
    x_bmode = conv_3d(bmode, num_filters//2, 5, activation='linear', padding='same', reuse=reuse, scope=scope + "_1_bmode",
                     weights_init=init)
    x_pd = conv_3d(pd, num_filters // 2, 5, activation='linear', padding='same', reuse=reuse, scope=scope + "_1_pd",
                weights_init=init)
    x = tflearn.layers.merge_ops.merge([x_bmode, x_pd], axis=4, mode='concat')
    x = conv_3d(x, num_filters, 5, activation='linear', padding='same', reuse=reuse,
                      scope=scope + "_1",
                      weights_init=init)
    x = tflearn.activation(x, "prelu")
    return x


def conv_block(input, num_convs, reuse, scope, init):
    x = input
    n_channels = int(x.get_shape()[-1])
    for i in range(num_convs):
        with tf.variable_scope("conv_" + str(i+1)):
            x = conv_3d(x, n_channels, filter_size=5, strides=1, activation='linear', padding='same',
                    weights_init=init,
                    scope=scope + "_2", reuse=reuse)
            if i == num_convs - 1:
                x = x + input
            # x = tflearn.layers.normalization.batch_normalization(x, reuse=reuse, scope=scope+"bn"+str(i))
            x = tflearn.activation(x, "prelu")
    return x


def conv_block2(input, feature, num_convs, reuse, init):
    x = tflearn.layers.merge_ops.merge([input, feature], axis=4, mode='concat')
    n_channels = int(input.get_shape()[-1])
    if num_convs == 1:
        with tf.variable_scope("conv_" + str(1)):
            x = conv_3d(x, n_channels*2, filter_size=5, strides=1, activation='linear', padding='same',
                        weights_init=init,
                        scope="conv_" + str(1), reuse=reuse)
            # x = tflearn.layers.normalization.batch_normalization(x, reuse=reuse, scope="bn_" + str(1))
            # input = tflearn.layers.normalization.batch_normalization(x, reuse=reuse, scope="bn_inp_" + str(1))
            input = x
            x = x + input
            # x = tflearn.layers.normalization.batch_normalization(x, reuse=reuse, scope="bn_merged_" + str(1))
            x = tflearn.activation(x, "prelu")
        return x

    with tf.variable_scope("conv_" + str(1)):
        x = conv_3d(x, n_channels * 2, filter_size=5, strides=1, activation='linear', padding='same',
                    weights_init=init,
                    scope="conv_" + str(1), reuse=reuse)
        # x = tflearn.layers.normalization.batch_normalization(x, reuse=reuse, scope="bn_" + str(1))
        x = tflearn.activation(x, "prelu")

    for i in range(1, num_convs):
        with tf.variable_scope("conv_" + str(i+1)):
            x = conv_3d(x, n_channels, filter_size=5, strides=1, activation='linear', padding='same',
                    weights_init=init,
                    scope="conv2_" + str(i+1), reuse=reuse)
            # x = tflearn.layers.normalization.batch_normalization(x, reuse=reuse, scope="bn_" + str(i+1))
            # input = tflearn.layers.normalization.batch_normalization(x, reuse=reuse, scope="bn_inp_" + str(i+1))
            input = x
            if i == num_convs - 1:
                x = x + input
            # x = tflearn.layers.normalization.batch_normalization(x, reuse=reuse, scope="bn_merged_" + str(i + 1))
            x = tflearn.activation(x, "prelu")
    return x


# Down convolutions with down sampling using strides
def down_block(input_tensor, reuse, scope, init):
    n_channels = int(input_tensor.get_shape()[-1])
    x = conv_3d(input_tensor, n_channels * 2, filter_size=2, strides=2, activation='linear', padding='same', weights_init=init,
                scope=scope + "_1_ds", reuse=reuse)
    # x = tflearn.layers.normalization.batch_normalization(x, reuse=reuse, scope=scope + "bn")
    x = tflearn.activation(x, "prelu")
    return x


# De-convolutions with concatenation
def up_block(input_tensor, concat, reuse, scope, init):
    n_channels = int(input_tensor.get_shape()[-1])
    x = conv_3d_transpose(input_tensor, n_channels, filter_size=2, strides=2,
                              output_shape=[concat.get_shape().as_list()[1]] * 3,
                              activation='linear', padding='same', weights_init=init, scope=scope + "_1",
                              reuse=reuse)
    # x = tflearn.layers.normalization.batch_normalization(x, reuse=reuse, scope=scope + "bn")
    x_out = tflearn.activation(x, "prelu")
    return x_out


class Model(object):
    def __init__(self, batch_size, reuse=False, tf_record_dir=None, num_epochs=0, weighting=[1] * 2):
        self.batch_size = batch_size
        record_reader = RecordReader(PowerDopplerProcessTup(data_loader))
        with tf.device('/cpu:0'):
            with tf.variable_scope("input"):
                if tf_record_dir:
                    x_shape = [-1] + list(segment_size_in) + [1]
                    y_shape = [-1] + list(segment_size_out) + [1]
                    if reuse:
                        X, Y, PD = record_reader.input_pipeline(False, batch_size, None, tf_record_dir)
                    else:
                        X, Y, PD = record_reader.input_pipeline(True, batch_size, num_epochs, tf_record_dir)
                    self.X = tf.reshape(X, x_shape)
                    self.PD = tf.reshape(PD, x_shape)
                    self.Y = tf.reshape(Y, y_shape)
                else:
                    self.X = tf.placeholder(
                        dtype=tf.float32,
                        shape=[None] + segment_size_in.tolist() + [1])
                    self.PD = tf.placeholder(
                        dtype=tf.float32,
                        shape=[None] + segment_size_in.tolist() + [1])
                    self.Y = tf.placeholder(
                        dtype=tf.float32,
                        shape=[None] + segment_size_out.tolist() + [1])
                X = self.X
                PD = self.PD
                Y = tf.cast(tf.one_hot(tf.reshape(tf.cast(self.Y, tf.uint8), [-1] + list(segment_size_out)), 2),
                            tf.float32)
                X_A = tf.split(X, 2)
                PD_A = tf.split(PD, 2)
                Y_A = tf.split(Y, 2)
        with tf.variable_scope("inference") as scope:
            losses = []
            preds = []
            for gpu_id in range(2):
                with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
                        if gpu_id == 0:
                            logits, self.feats = self.build_net(X_A[gpu_id], PD_A[gpu_id], False)
                        else:
                            logits, self.feats = self.build_net(X_A[gpu_id], PD_A[gpu_id], True)
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
        return PowerDopplerFullInferer(segment_size_in, segment_size_out, crop_by, stride, self.batch_size)

    def build_feats_writer(self):
        crop_by = 2
        stride_test = segment_size_in - crop_by
        segment_size_out = stride_test
        return StandardFeatsWriter(segment_size_in, segment_size_out, crop_by, stride_test, self.batch_size, 16)

    # def build_net(self, X, PD, reuse=False):
    #     init = tflearn.initializations.xavier()
    #     num_convolutions = (1, 2, 3, 3)
    #     num_levels = 3
    #     # Encoder path
    #     with tf.variable_scope("input_level"):
    #         net = input_block(X, PD, 8, reuse, "input_level", init)
    #
    #     features = list()
    #
    #     # Down convolutions
    #     for level in range(num_levels):
    #         with tf.variable_scope("level_" + str(level + 1)):
    #             net = conv_block(net, num_convolutions[level], reuse, "level_" + str(level + 1), init)
    #             features.append(net)
    #             with tf.variable_scope("down_conv"):
    #                 net = down_block(net, reuse, "down_conv" + str(level + 1), init)
    #
    #     # Bottom level
    #     with tf.variable_scope("bottom"):
    #         net = conv_block(net, 3, reuse, "bottom", init)
    #
    #     # Decoder path
    #     for level in reversed(range(num_levels)):
    #         with tf.variable_scope("up_level_" + str(level + 1)):
    #             f = features[level]
    #             with tf.variable_scope("up_conv"):
    #                 net = up_block(net, f, reuse, "up_conv" + str(level + 1), init)
    #
    #             net = conv_block2(net, f, num_convolutions[level], reuse, init)
    #
    #     # Output level
    #     with tf.variable_scope("out"):
    #         logits = tflearn.layers.conv_3d(net, 8, 1, activation='linear', padding='same', regularizer='L2',
    #                                          reuse=reuse, scope='fc1', weights_init=init)
    #         logits = tflearn.layers.normalization.batch_normalization(logits, reuse=reuse, scope='bn_out')
    #         logits = tflearn.layers.conv_3d(logits, 2, 1, activation='linear', padding='same', regularizer='L2',
    #                                          reuse=reuse, scope='output', weights_init=init)
    #     return logits, net
    def build_net(self, X, PD, reuse=False, segment_size_in=segment_size_in,feats=None):
        # Using TFLearn wrappers for network building
        init = tflearn.initializations.xavier()
        with tf.variable_scope("level1", reuse=reuse):
            net1_bmode = tflearn.layers.conv_3d(X, 8, 5, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='conv1-1-bmode', weights_init=init)
            net1_pd = tflearn.layers.conv_3d(PD, 8, 5, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='conv1-1-pd', weights_init=init)
            net1 = tflearn.layers.merge_ops.merge([net1_bmode, net1_pd], axis=4, mode='concat')
            net1 = tflearn.layers.conv_3d(net1, 16, 5, activation='linear', padding='same', regularizer='L2',
                                          reuse=reuse, scope='conv1-1', weights_init=init)
            net1 = tflearn.activation(net1, 'prelu')

        with tf.variable_scope("level2", reuse=reuse):
            net2_in = tflearn.layers.conv_3d(net1, 32, 2, strides=2, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='ds1', weights_init=init)
            # net2_in = tflearn.layers.normalization.batch_normalization(net2_in, reuse=reuse, scope='batch2-1')
            net2 = tflearn.layers.conv_3d(net2_in, 32, 5, activation='linear',padding='same', regularizer='L2',reuse=reuse,scope='conv2-1', weights_init=init)
            net2 = tflearn.layers.conv_3d(net2, 32, 5, activation='linear',padding='same', regularizer='L2',reuse=reuse,scope='conv2-2', weights_init=init)
            # net2 = tflearn.layers.normalization.batch_normalization(net2, reuse=reuse, scope='batch2-2')
            net2 = tflearn.activation(net2, 'prelu')

        with tf.variable_scope("level3"):
            net3_in = tflearn.layers.conv_3d(net2, 64, 2, strides=2, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='ds2', weights_init=init)
            # net3_in = tflearn.layers.normalization.batch_normalization(net3_in, reuse=reuse, scope='batch3-1')
            net3 = tflearn.layers.conv_3d(net3_in, 64, 5, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='conv3-1', weights_init=init)
            net3 = tflearn.layers.conv_3d(net3, 64, 5, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='conv3-2', weights_init=init)
            net3 = tflearn.layers.conv_3d(net3, 64, 5, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='conv3-3', weights_init=init)
            # net3 = tflearn.layers.normalization.batch_normalization(net3, reuse=reuse, scope='batch3-2')
            net3 = tflearn.activation(net3, 'prelu')

        with tf.variable_scope("level4"):
            net4_in = tflearn.layers.conv_3d(net3, 128, 2, strides=2, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='ds3', weights_init=init)
            # net4_in = tflearn.layers.normalization.batch_normalization(net4_in, reuse=reuse, scope='batch4-1')
            net4 = tflearn.layers.conv_3d(net4_in, 128, 5, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='conv4-1', weights_init=init)
            net4 = tflearn.layers.conv_3d(net4, 128, 5, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='conv4-2', weights_init=init)
            net4 = tflearn.layers.conv_3d(net4, 128, 5, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='conv4-3', weights_init=init)
            # net4 = tflearn.layers.normalization.batch_normalization(net4, reuse=reuse, scope='batch4-2')
            net4 = tflearn.activation(net4, 'prelu')

        with tf.variable_scope("bottom"):
            netbot_in = tflearn.layers.conv_3d(net4, 256, 2, strides=2, activation='prelu',padding='same',regularizer='L2',reuse=reuse,scope='ds3', weights_init=init)
            # net4_in = tflearn.layers.normalization.batch_normalization(net4_in, reuse=reuse, scope='batch4-1')
            netbot = tflearn.layers.conv_3d(netbot_in, 256, 5, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='convbot-1', weights_init=init)
            netbot = tflearn.layers.conv_3d(netbot, 256, 5, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='convbot-2', weights_init=init)
            netbot = tflearn.layers.conv_3d(netbot, 256, 5, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='convbot-3', weights_init=init)
            # net4 = tflearn.layers.normalization.batch_normalization(net4, reuse=reuse, scope='batch4-2')
            netbot = tflearn.activation(netbot, 'prelu')

        with tf.variable_scope("level4_up"):
            net4_up = tflearn.layers.conv.conv_3d_transpose(netbot, 128, 2, [net4.get_shape().as_list()[1]] * 3,
                                                            strides=2, activation='prelu', padding='same',
                                                            regularizer='L2', reuse=reuse, scope='trans1',
                                                            weights_init=init)
            net4_up = tflearn.layers.merge_ops.merge([net4_up, net4], 'elemwise_sum', name='merge5')
            net4_up = tflearn.layers.conv_3d(net4_up, 128, 5, activation='linear', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv5-1', weights_init=init)
            net4_up = tflearn.layers.conv_3d(net4_up, 128, 5, activation='linear', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv5-2', weights_init=init)
            net4_up = tflearn.layers.conv_3d(net4_up, 128, 5, activation='linear', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv5-3', weights_init=init)
            # net4_up = tflearn.layers.normalization.batch_normalization(net4_up, reuse=reuse, scope='batch5-2')
            net4_up = tflearn.activation(net4_up, 'prelu')

        with tf.variable_scope("level3_up"):
            net3_up = tflearn.layers.conv.conv_3d_transpose(net4_up, 64, 2, [net3.get_shape().as_list()[1]] * 3,
                                                            strides=2, activation='prelu', padding='same',
                                                            regularizer='L2', reuse=reuse, scope='trans2',
                                                            weights_init=init)
            net3_up = tflearn.layers.merge_ops.merge([net3_up, net3], 'elemwise_sum', name='merge5')
            net3_up = tflearn.layers.conv_3d(net3_up, 64, 5, activation='linear', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv6-1', weights_init=init)
            net3_up = tflearn.layers.conv_3d(net3_up, 64, 5, activation='linear', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv6-2', weights_init=init)
            net3_up = tflearn.layers.conv_3d(net3_up, 64, 5, activation='linear', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv6-3', weights_init=init)
            # net3_up = tflearn.layers.normalization.batch_normalization(net3_up, reuse=reuse, scope='batch5-2')
            net3_up = tflearn.activation(net3_up, 'prelu')

        with tf.variable_scope("level2_up"):
            net2_up = tflearn.layers.conv.conv_3d_transpose(net3_up, 32, 2, [net2.get_shape().as_list()[1]] * 3,
                                                            strides=2, activation='prelu', padding='same',
                                                            regularizer='L2', reuse=reuse, scope='trans3',
                                                            weights_init=init)
            net2_up = tflearn.layers.merge_ops.merge([net2_up, net2], 'elemwise_sum', name='merge5')
            net2_up = tflearn.layers.conv_3d(net2_up, 32, 5, activation='linear', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv7-1', weights_init=init)
            net2_up = tflearn.layers.conv_3d(net2_up, 32, 5, activation='linear', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv7-2', weights_init=init)
            # net2_up = tflearn.layers.normalization.batch_normalization(net2_up, reuse=reuse, scope='batch5-2')
            net2_up = tflearn.activation(net2_up, 'prelu')

        with tf.variable_scope("level1_up"):
            net1_up = tflearn.layers.conv.conv_3d_transpose(net2_up, 16, 2, [net1.get_shape().as_list()[1]] * 3,
                                                            strides=2, activation='prelu', padding='same',
                                                            regularizer='L2', reuse=reuse, scope='trans3',
                                                            weights_init=init)
            net1_up = tflearn.layers.merge_ops.merge([net1_up, net1], 'elemwise_sum', name='merge5')
            net1_up = tflearn.layers.conv_3d(net1_up, 16, 5, activation='linear', padding='same', regularizer='L2',
                                             reuse=reuse, scope='conv8-1', weights_init=init)
            # net1_up = tflearn.layers.normalization.batch_normalization(net1_up, reuse=reuse, scope='batch5-2')
            net1_up = tflearn.activation(net1_up, 'prelu')

        with tf.variable_scope("out"):
            net_fc1 = tflearn.layers.conv_3d(net1_up, 16, 1, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='fc1', weights_init=init)
            net_fc1 = tflearn.layers.normalization.batch_normalization(net_fc1, reuse=reuse, scope='batch_fc_1')
            net_fc1 = tflearn.activation(net_fc1, 'prelu')
            net_out = tflearn.layers.conv_3d(net_fc1, 2, 1, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='output',weights_init=init)

        return net_out, net1_bmode
