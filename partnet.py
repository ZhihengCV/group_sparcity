"""Part selection model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def _feature_map_image_summary(x):
    tensor_list = [tf.expand_dims(tensor, 3)  for tensor in tf.unpack(x, axis=3)]
    for idx, tensor in enumerate(tensor_list):
        tensor_name = x.op.name + '/active_%d' % idx
        tf.image_summary(tensor_name, tensor, max_images=2 )


def _one_dimention_image_summery(x, slice_num=2):
    tensor_reshape = tf.reshape(tf.shape(x)[0], -1, slice_num**2, 1)
    tf.image_summary(x.op.name, tensor_reshape, max_images=2)


def slice_patch(input_, slice_num=2, name='slice_patch'):
    """slice feature map into part"""
    with tf.name_scope(name):
        in_shape = input_.get_shape()
        padding_h = in_shape[1] % 2
        padding_w = in_shape[2] % 2
        if padding_h or padding_w:
            input_ = tf.pad(input_,
                            [[0, 0], [padding_h, 0], [padding_w, 0], [0, 0]],
                            "CONSTANT")
            in_shape = input_.get_shape()
        h_stride = in_shape[1] / 2
        w_stride = in_shape[2] / 2
        patch_list = []
        for i in range(slice_num):
            for j in range(slice_num):
                part = tf.slice(input_, [0, i * h_stride, j * w_stride, 0], [-1, h_stride, w_stride, -1])
                patch_list.append(part)
    return patch_list


def merge_patch(patch_list, name='merge_patch'):
    """merge different feature map into one vector"""
    with tf.name_scope(name):
        s_loss = tf.Variable(0.0, name='sparse_loss')
        for input_ in patch_list:
            assert len(list(input_)) == 2, "concat dimension should equal to two"
            input_square = tf.square(input_)
            patch_sum = tf.sqrt(tf.reduce_sum(input_square))
            s_loss += patch_sum
        tf.add_to_collection("SPARSE_LOSS", s_loss)
        return tf.concat(1, patch_list, name='combined_feature')


def partnet_v1(images, num_classes=10, is_training=False,
               dropout_keep_prob=0.5, scope='PartNet', slice_num=2):
    """This version of part_net does not have full image path"""

    with tf.variable_scope(scope, 'PartNet', [images, num_classes]):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            trainable=is_training):
            basenet = slim.conv2d(images, 32, [3, 3], scope='conv1')  # default is same padding and relu
            patch_list = []
            for num, part in enumerate(slice_patch(basenet, slice_num)):
                with tf.name_scope("path_%d" % num):
                    part_net = slim.conv2d(part, 32, [5, 5], scope='conv2_%d' % num)
                    _activation_summary(part_net)
                    _feature_map_image_summary(part_net)
                    part_net = slim.max_pool2d(part_net, [2, 2], 2, scope='pool2_%d' % num)
                    part_net = slim.flatten(part_net, scope='flatten_%d')
                    part_net = slim.fully_connected(part_net, 64, scope='fc3_%d' % num)
                    patch_list.append(part_net)
            basenet = merge_patch(patch_list)
            _activation_summary(basenet)
            _one_dimention_image_summery(basenet, slice_num=2)
            basenet = slim.fully_connected(basenet, 128, scope='fc4')
            basenet = slim.dropout(basenet, dropout_keep_prob, is_training=is_training,
                                   scope='dropout4')
            basenet = slim.fully_connected(basenet, num_classes,
                                           biases_initializer=tf.zeros_initializer,
                                           weights_initializer=trunc_normal(1 / 128.0),
                                           weights_regularizer=None,
                                           activation_fn=None,
                                           scope='logits')
            return basenet
