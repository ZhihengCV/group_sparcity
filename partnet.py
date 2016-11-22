"""Part selection model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)

INITIAL_LEARNING_RATE = 0.01
FLAGS = tf.app.flags.FLAGS
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 55000


def _activation_summary(x):
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def _feature_map_image_summary(x):
    tensor_list = [tf.expand_dims(tensor, 3) for tensor in tf.unpack(x, axis=3)]
    for idx, tensor in enumerate(tensor_list):
        tensor_name = x.op.name + '/active_%d' % idx
        tf.image_summary(tensor_name, tensor, max_images=2)


def _one_dimention_image_summery(x, slice_num=2):
    tensor_reshape = tf.reshape(x, [tf.shape(x)[0], -1, slice_num ** 2, 1])
    tf.image_summary(x.op.name, tensor_reshape, max_images=1)


def _slice_patch(input_, slice_num=2, name='slice_patch'):
    """helper function slice feature map into part"""
    with tf.name_scope(name):
        in_shape = input_.get_shape()
        padding_h = in_shape[1] % 2
        padding_w = in_shape[2] % 2
        # Pay attention here
        if (padding_h > 0) or (padding_w > 0):
            input_ = tf.pad(input_,
                            [[0, 0], [padding_h.value, 0], [padding_w.value, 0], [0, 0]],
                            "CONSTANT")
            in_shape = input_.get_shape()
        h_stride = int(in_shape[1].value / 2)
        w_stride = int(in_shape[2].value / 2)
        patch_list = []
        for i in range(slice_num):
            for j in range(slice_num):
                part = tf.slice(input_, [0, i * h_stride, j * w_stride, 0], [-1, h_stride, w_stride, -1])
                patch_list.append(part)
    return patch_list


def _merge_patch(patch_list, name='merge_patch'):
    """helper function merge different feature map into one vector"""
    with tf.name_scope(name):
        patch_loss_list = []
        for input_ in patch_list:
            assert input_.get_shape().ndims == 2, "concat dimension should equal to two"
            input_square = tf.square(input_)
            patch_loss = tf.sqrt(tf.reduce_sum(input_square, 1))
            patch_loss_list.append(patch_loss)
        sparse_loss = tf.reduce_mean(tf.add_n(patch_loss_list), name='sparse_loss')
        tf.add_to_collection('sparse_loss', sparse_loss)
        return tf.concat(1, patch_list, name='combined_feature')


def inference(images, num_classes=10, is_training=False,
              dropout_keep_prob=0.5, scope='PartNet', slice_num=2):
    """This version of part_net does not have full image path"""
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            basenet = slim.conv2d(images, 32, [3, 3], scope='conv1')  # default is same padding and relu
            patch_list = []
            for num, part in enumerate(_slice_patch(basenet, slice_num)):
                with tf.variable_scope("path_%d" % num):
                    part_net = slim.conv2d(part, 32, [5, 5], scope='conv2_%d' % num)
                    _activation_summary(part_net)
                    _feature_map_image_summary(part_net)
                    part_net = slim.max_pool2d(part_net, [2, 2], 2, scope='pool2_%d' % num)
                    part_net = slim.flatten(part_net, scope='flatten_%d' % num)
                    part_net = slim.fully_connected(part_net, 64, scope='fc3_%d' % num)
                    patch_list.append(part_net)
            basenet = _merge_patch(patch_list)
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


def loss(logits, labels, sparse_weight=0.1):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

    Returns:
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    with tf.name_scope("loss"):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='cross_entropy_per_example')
        softmax_loss = tf.reduce_mean(cross_entropy, name='softmax_loss')
        tf.scalar_summary('softmax_loss', softmax_loss)
        regularization_loss = tf.add_n(slim.losses.get_regularization_losses(),
                                       name="regular_loss")
        tf.scalar_summary('regular_loss', regularization_loss)
        sparse_loss = tf.get_collection('sparse_loss')[0]
        tf.scalar_summary('sparse_loss_origin', sparse_loss)
        sparse_loss_w = tf.mul(sparse_loss, sparse_weight, name='sparse_loss_weighted')
        tf.scalar_summary('sparse_loss_weighted', sparse_loss_w)
        total_loss = tf.add_n([softmax_loss, regularization_loss, sparse_loss_w],
                              name='total_loss')
        tf.scalar_summary('total_loss', total_loss)
        return total_loss


def train(total_loss, global_step):
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables.

    Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
    Returns:
    train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)

    # Decay the learning rate exponentially based on the number of steps.
    boundaries = [10 * num_batches_per_epoch, 20 * num_batches_per_epoch]
    values = [0.01, 0.003, 0.001]
    lr = tf.train.piecewise_constant(global_step, boundaries, values)
    tf.scalar_summary('learning_rate', lr)
    optimizer = tf.train.MomentumOptimizer(lr, 0.99)
    grads = optimizer.compute_gradients(total_loss)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
