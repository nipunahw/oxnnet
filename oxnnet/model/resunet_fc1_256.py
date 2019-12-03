import os
import numpy as np
import tensorflow as tf
import tflearn

from oxnnet.data_loader import StandardDataLoader
from oxnnet.record import RecordWriter, StandardProcessTup, RecordReader
from oxnnet.full_inferer import StandardFullInferer
from oxnnet.feats_writer import StandardFeatsWriter

segment_size_in_test = np.array([64]*3)
segment_size_in = np.array([64]*3)
segment_size_out = segment_size_in #calc_out_shape(segment_size_in) #-16
crop_by = 0 #(segment_size_in-segment_size_out)/2
train_eval_test_no = [600,50,600]
stride = np.array([32]*3,dtype=np.int)
stride_test = np.array([32]*3,dtype=np.int)
batch_size_test = 10

data_loader = StandardDataLoader(stride, segment_size_in)

def build_record_writer(data_dir, dir_type_flag):
    if dir_type_flag == 'meta':
        data_loader.read_metadata(data_dir)
    elif dir_type_flag == 'deepmedic':
        data_loader.read_deepmedic_dir(data_dir)
    else:
        data_loader.read_data_dir(data_dir, train_eval_test_no)
    return RecordWriter(data_loader, StandardProcessTup, num_of_threads=os.cpu_count())

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
                # Y = tf.cast(self.Y, tf.float32)
                Y = tf.cast(tf.one_hot(tf.reshape(tf.cast(self.Y, tf.uint8), [-1] + list(segment_size_out)), 2),
                            tf.float32)
                X_A = tf.split(X, 2)
                Y_A = tf.split(Y, 2)

        with tf.variable_scope("inference") as scope:
            # if tf_record_dir:
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
                            losses.append(loss_op)
                        # Choose the metrics to compute:
                        names_to_values, names_to_updates = tf.contrib.metrics.aggregate_metric_map({
                            'accuracy': tf.contrib.metrics.streaming_accuracy(softmax_logits, Y_A[gpu_id]),
                            'precision': tf.contrib.metrics.streaming_precision(softmax_logits, Y_A[gpu_id]),
                            'recall': tf.contrib.metrics.streaming_recall(softmax_logits, Y_A[gpu_id]),
                            'mse': tf.contrib.metrics.streaming_mean_squared_error(softmax_logits, Y_A[gpu_id]),
                        })
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
        return StandardFullInferer(segment_size_in_test, segment_size_out, crop_by, stride_test, self.batch_size)

    def build_feats_writer(self):
        crop_by = 2
        stride_test = segment_size_in - crop_by 
        segment_size_out = stride_test
        return StandardFeatsWriter(segment_size_in_test, segment_size_out, crop_by, stride_test, self.batch_size, 16)

    def build_net(self, X, reuse=False, segment_size_in=segment_size_in,feats=None):
        # Using TFLearn wrappers for network building
        init = tflearn.initializations.variance_scaling()
        with tf.variable_scope("level1"):
            net1 = tflearn.layers.conv_3d(X, 16, 3, activation='linear', padding='same', regularizer='L2', reuse=reuse,
                                          scope='conv1-1', weights_init=init)
            net1 = tflearn.activation(net1, 'prelu')

        with tf.variable_scope("level2"):
            net2_in = tflearn.layers.conv_3d(net1, 32, 2, strides=2, activation='prelu', padding='same',
                                             regularizer='L2', reuse=reuse, scope='ds1', weights_init=init)
            net2 = tflearn.layers.conv_3d(net2_in, 32, 3, activation='linear', padding='same', regularizer='L2',
                                          reuse=reuse, scope='conv2-1', weights_init=init)
            net2 = tflearn.layers.merge_ops.merge([net2, net2_in], 'elemwise_sum', name='merge2')
            net2 = tflearn.activation(net2, 'prelu')

        with tf.variable_scope("level3"):
            net3_in = tflearn.layers.conv_3d(net2, 64, 2, strides=2, activation='prelu', padding='same',
                                             regularizer='L2', reuse=reuse, scope='ds2', weights_init=init)
            net3 = tflearn.layers.conv_3d(net3_in, 64, 3, activation='linear', padding='same', regularizer='L2',
                                          reuse=reuse, scope='conv3-1', weights_init=init)
            net3 = tflearn.layers.merge_ops.merge([net3, net3_in], 'elemwise_sum', name='merge3')
            net3 = tflearn.activation(net3, 'prelu')

        with tf.variable_scope("level4"):
            net4_in = tflearn.layers.conv_3d(net3, 128, 2, strides=2, activation='prelu', padding='same',
                                             regularizer='L2', reuse=reuse, scope='ds3', weights_init=init)
            net4 = tflearn.layers.conv_3d(net4_in, 128, 3, activation='linear', padding='same', regularizer='L2',
                                          reuse=reuse, scope='conv4-1', weights_init=init)
            net4 = tflearn.layers.merge_ops.merge([net4, net4_in], 'elemwise_sum', name='merge4')
            net4 = tflearn.activation(net4, 'prelu')

        with tf.variable_scope("level5"):
            net5_in = tflearn.layers.conv_3d(net4, 256, 2, strides=2, activation='prelu', padding='same',
                                             regularizer='L2', reuse=reuse, scope='ds4', weights_init=init)
            net5 = tflearn.layers.conv_3d(net5_in, 256, 3, activation='linear', padding='same', regularizer='L2',
                                          reuse=reuse, scope='conv5-1', weights_init=init)
            net5 = tflearn.layers.merge_ops.merge([net5, net5_in], 'elemwise_sum', name='merge5')
            net5 = tflearn.activation(net5, 'prelu')

        with tf.variable_scope("level6"):
            net6 = tflearn.layers.conv.conv_3d_transpose(net5, 128, 2, (segment_size_in // 8).tolist(), strides=2,
                                                         activation='prelu', padding='same', regularizer='L2',
                                                         reuse=reuse, scope='trans1', weights_init=init)
            net6_in = tf.concat([net6, net4], axis=4)  # =tf.slice(net11,begin,tf.shape(net15))
            net6 = tflearn.layers.conv_3d(net6_in, 256, 3, activation='linear', padding='same', regularizer='L2',
                                          reuse=reuse, scope='conv6-1', weights_init=init)
            net6 = tflearn.layers.merge_ops.merge([net6, net6_in], 'elemwise_sum', name='merge6')
            net6 = tflearn.activation(net6, 'prelu')

        with tf.variable_scope("level7"):
            net7 = tflearn.layers.conv.conv_3d_transpose(net6, 64, 2, (segment_size_in // 4).tolist(), strides=2,
                                                         activation='prelu', padding='same', regularizer='L2',
                                                         reuse=reuse, scope='trans2', weights_init=init)
            net7_in = tf.concat([net7, net3], axis=4)  # tf.slice(net8, begin, tf.shape(net18))
            net7 = tflearn.layers.conv_3d(net7_in, 128, 3, activation='linear', padding='same', regularizer='L2',
                                          reuse=reuse, scope='conv7-1', weights_init=init)
            net7 = tflearn.layers.merge_ops.merge([net7, net7_in], 'elemwise_sum', name='merge7')
            net7 = tflearn.activation(net7, 'prelu')

        with tf.variable_scope("level8"):
            net8 = tflearn.layers.conv.conv_3d_transpose(net7, 32, 2, (segment_size_in // 2).tolist(), strides=2,
                                                         activation='prelu', padding='same', regularizer='L2',
                                                         reuse=reuse, scope='trans3', weights_init=init)
            net8_in = tf.concat([net8, net2], axis=4)  # tf.slice(net5, begin, tf.shape(net18))
            net8 = tflearn.layers.conv_3d(net8_in, 64, 3, activation='linear', padding='same', regularizer='L2',
                                          reuse=reuse, scope='conv8-1', weights_init=init)
            net8 = tflearn.layers.merge_ops.merge([net8, net8_in], 'elemwise_sum', name='merge8')
            net8 = tflearn.activation(net8, 'prelu')

        with tf.variable_scope("level9"):
            net9 = tflearn.layers.conv.conv_3d_transpose(net8, 16, 2, segment_size_in.tolist(), strides=2,
                                                         activation='prelu', padding='same', regularizer='L2',
                                                         reuse=reuse, scope='trans4', weights_init=init)
            net9_in = tf.concat([net9, net1], axis=4)  # tf.slice(net2, begin, tf.shape(net25))
            net9 = tflearn.layers.conv_3d(net9_in, 32, 3, activation='linear', padding='same', regularizer='L2',
                                          reuse=reuse, scope='conv9-1', weights_init=init)
            net9 = tflearn.layers.merge_ops.merge([net9, net9_in], 'elemwise_sum', name='merge9')
            net9 = tflearn.activation(net9, 'prelu')

        with tf.variable_scope("out"):
            net_fc1 = tflearn.layers.conv_3d(net9, 32, 1, activation='linear', padding='same', regularizer='L2',
                                             reuse=reuse, scope='fc1', weights_init=init)
            net_fc1 = tflearn.activation(net_fc1, 'prelu')
            net_fc1 = tflearn.layers.core.dropout(net_fc1, 0.5)
            net_out = tflearn.layers.conv_3d(net_fc1, 1, 1, activation='linear', padding='same', regularizer='L2',
                                             reuse=reuse, scope='output', weights_init=init)

        return net_out, net_fc1
