import os
import numpy as np
import tensorflow as tf
import tflearn

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


#https://github.com/caglar/noisy_units/blob/master/codes/tf/nunits.py
HardTanh = lambda x: tf.minimum(tf.maximum(x, -1.), 1.)
lin_sigmoid = lambda x: 0.25 * x + 0.5
# Sigmoid = lambda x, use_noise=0: T.nnet.sigmoid(x)
HardSigmoid = lambda x, angle=0.25: tf.maximum(tf.minimum(angle*x + 0.5, 1.0), 0.0)
HardSigmoid = lambda x: tf.minimum(tf.maximum(lin_sigmoid(x), 0.), 1.)

def NTanh(x, use_noise, alpha=1.05, c=0.5, half_normal=False):
    """
    Noisy Hard Tanh Units: NAN without learning p
    ----------------------------------------------------
    Arguments:
        x: tensorflow tensor variable, input of the function.
        use_noise: bool, whether to add noise or not to the activations, this is in particular
        useful for the test time, in order to disable the noise injection.
        c: float, standard deviation of the noise
        alpha: the leaking rate from the linearized function to the nonlinear one.
    """
    threshold = 1.0
    signs = tf.sign(x)
    delta = tf.abs(x) - threshold

    scale = c * (tf.sigmoid(delta**2) - 0.5)**2
    if alpha > 1.0 and  half_normal:
           scale *= -1.0
    zeros=tf.zeros(tf.shape(x), dtype=tf.float32, name=None)
    def noise_func() :return tf.abs(tf.random_normal(tf.shape(x), mean=0.0, stddev=1.0, dtype=tf.float32))
    def zero_func (): return zeros+ 0.797  if half_normal   else zeros
    noise=tf.cond(use_noise,noise_func,zero_func)

    eps = scale * noise + alpha * delta
    z = x - signs * eps
    test=tf.cast(tf.greater_equal(tf.abs(x) , threshold),tf.float32)
    res = test * z + (1. - test) *  HardTanh(x)
    return res

def build_record_writer(data_dir, dir_type_flag):
    data_loader.read_data_dir(data_dir, train_eval_test_no)
    return RecordWriter(data_loader, PowerDopplerProcessTup, num_of_threads=os.cpu_count() - 1)

class Model(object):
    def __init__(self, batch_size, reuse=False, tf_record_dir=None, num_epochs=0, weighting=[1]*2):
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

    def conv_layer(self, input_tensor, nb_filter, stage, reuse, init, merge=False):
        x = tflearn.layers.conv_3d(input_tensor, nb_filter, 3, padding='same', regularizer='L2',
                                     reuse=reuse, scope='ds'+stage, weights_init=init)
        if not merge:
            x = tflearn.layers.conv_3d(x, nb_filter, 3, padding='same', regularizer='L2',
                                          reuse=reuse, scope='conv'+stage, weights_init=init)
        x = NTanh(x, tf.constant(True, dtype=tf.bool))
        return x

    def deconv_layer(self, input_tensor, concat, nb_filter, stage, reuse, init):
        x = tflearn.layers.conv.conv_3d_transpose(input_tensor, nb_filter, 2, [concat.get_shape().as_list()[1]] * 3,
                                                  strides=2, padding='same',
                                                  regularizer='L2', reuse=reuse, scope='trans'+stage, weights_init=init)
        return x

    def build_net(self, X, PD, reuse=False):
        # Using TFLearn wrappers for network building
        init = tflearn.initializations.xavier()
        with tf.variable_scope("level0_0", reuse=reuse):
            net0_0 = self.conv_layer(X, 4, '0_0', reuse, init)
            net0_0_pd = self.conv_layer(PD, 4, '0_0_pd', reuse, init)
            net0_0_merged = tflearn.layers.merge_ops.merge([net0_0, net0_0_pd], axis=4, mode='concat')
            pool0_0 = tflearn.layers.conv.max_pool_3d(net0_0_merged, 2, strides=2)

        with tf.variable_scope("level1_0", reuse=reuse):
            net1_0 = self.conv_layer(pool0_0, 16, '1_0', reuse, init)
            pool1_0 = tflearn.layers.conv.max_pool_3d(net1_0, 2, strides=2)

        with tf.variable_scope("level0_1", reuse=reuse):
            up0_1 = self.deconv_layer(net1_0, net0_0, 8, '1_2', reuse, init)
            net0_1 = tflearn.layers.merge_ops.merge([up0_1, net0_0], axis=4, mode='concat')
            net0_1 = self.conv_layer(net0_1, 8, '1_2', reuse, init, True)

        with tf.variable_scope("level2_0"):
            net2_0 = self.conv_layer(pool1_0, 32, '4', reuse, init)
            pool2_0 = tflearn.layers.conv.max_pool_3d(net2_0, 2, strides=2)

        with tf.variable_scope("level1_1", reuse=reuse):
            up1_1 = self.deconv_layer(net2_0, net1_0, 16, '2_2', reuse, init)
            net1_1 = tflearn.layers.merge_ops.merge([up1_1, net1_0], axis=4, mode='concat')
            net1_1 = self.conv_layer(net1_1, 16, '2_2', reuse, init, True)

        with tf.variable_scope("level0_2", reuse=reuse):
            up0_2 = self.deconv_layer(net1_1, net0_1, 8, '1_3', reuse, init)
            net0_2 = tflearn.layers.merge_ops.merge([up0_2, net0_0, net0_1], axis=4, mode='concat')
            net0_2 = self.conv_layer(net0_2, 8, '1_3', reuse, init, True)

        with tf.variable_scope("level3_0"):
            net3_0 = self.conv_layer(pool2_0, 64, '3_0', reuse, init)
            pool3_0 = tflearn.layers.conv.max_pool_3d(net3_0, 2, strides=2)

        with tf.variable_scope("level2_1", reuse=reuse):
            up2_1 = self.deconv_layer(net3_0, net2_0, 32, '2_1', reuse, init)
            net2_1 = tflearn.layers.merge_ops.merge([up2_1, net2_0], axis=4, mode='concat')
            net2_1 = self.conv_layer(net2_1, 32, '2_1', reuse, init, True)

        with tf.variable_scope("level1_2", reuse=reuse):
            up1_2 = self.deconv_layer(net2_1, net1_1, 16, '1_2', reuse, init)
            net1_2 = tflearn.layers.merge_ops.merge([up1_2, net1_0, net1_1], axis=4, mode='concat')
            net1_2 = self.conv_layer(net1_2, 16, '1_2', reuse, init, True)

        with tf.variable_scope("level0_3", reuse=reuse):
            up0_3 = self.deconv_layer(net1_2, net0_2, 8, '0_3', reuse, init)
            net0_3 = tflearn.layers.merge_ops.merge([up0_3, net0_0, net0_1, net0_2], axis=4, mode='concat')
            net0_3 = self.conv_layer(net0_3, 8, '0_3', reuse, init, True)

        with tf.variable_scope("level4_0"):
            net4_0 = self.conv_layer(pool3_0, 128, '4_0', reuse, init)

        with tf.variable_scope("level3_1"):
            up3_1 = self.deconv_layer(net4_0, net3_0, 64, '3_1', reuse, init)
            net3_1 = tflearn.layers.merge_ops.merge([up3_1, net3_0], axis=4, mode='concat')
            net3_1 = self.conv_layer(net3_1, 64, '3_1', reuse, init, True)

        with tf.variable_scope("level2_2"):
            up2_2 = self.deconv_layer(net3_1, net2_1, 32, '2_2', reuse, init)
            net2_2 = tflearn.layers.merge_ops.merge([up2_2, net2_0, net2_1], axis=4, mode='concat')
            net2_2 = self.conv_layer(net2_2, 32, '2_2', reuse, init, True)

        with tf.variable_scope("level1_3"):
            up1_3 = self.deconv_layer(net2_2, net1_2, 16, '1_3', reuse, init)
            net1_3 = tflearn.layers.merge_ops.merge([up1_3, net1_0, net1_1, net1_2], axis=4, mode='concat')
            net1_3 = self.conv_layer(net1_3, 16, '1_3', reuse, init, True)

        with tf.variable_scope("level0_4"):
            up0_4 = self.deconv_layer(net1_3, net0_3, 8, '0_4', reuse, init)
            net0_4 = tflearn.layers.merge_ops.merge([up0_4, net0_0, net0_1, net0_2, net0_3], axis=4, mode='concat')
            net0_4 = self.conv_layer(net0_4, 8, '0_4', reuse, init, True)

        with tf.variable_scope("out"):
            net_fc1 = tflearn.layers.conv_3d(net0_4, 8, 1, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='fc1', weights_init=init)
            net_fc1 = tflearn.layers.normalization.batch_normalization(net_fc1, reuse=reuse, scope='batch_fc_1')
            net_fc1 = NTanh(net_fc1, tf.constant(True, dtype=tf.bool))
            net_out = tflearn.layers.conv_3d(net_fc1, 2, 1, activation='linear',padding='same',regularizer='L2',reuse=reuse,scope='output',weights_init=init)

        return net_out, net0_0
