import tensorflow as tf
tf.disable_v2_behavior()

from tensorflow.layers import flatten
import tensorflow.contrib.slim as slim


def conv2d_PC(input_, output_dim, ks=3, s=1, stddev=0.02, padding='SAME', name='conv2d'):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)


def batch_norm(x, is_training, name='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                        is_training=is_training, scope=name)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def bottleneck_layer(x, isTrain, filters, scope):
    # print(x)
    with tf.name_scope(scope):
        x = batch_norm(x, is_training=isTrain, name=scope + '_batch1')
        # print("bn: ", x)
        x = tf.nn.relu(x)
        x = conv2d_PC(x, filters * 2, ks=1, name=scope + '_conv1')
        x = tf.layers.dropout(x, 0.5, isTrain)
        # print("conv: ", x)
        x = batch_norm(x, is_training=isTrain, name=scope + '_batch2')
        # print("bn: ", x)
        x = tf.nn.relu(x)
        x = conv2d_PC(x, filters / 2, ks=3, name=scope + '_conv2')
        x = tf.layers.dropout(x, 0.5, isTrain)
        # print("conv: ", x)
        return x


def dense_block(input_x, nb_layers, layer_name, isTrain, filters):
    with tf.name_scope(layer_name) as scope:
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer(input_x, isTrain, filters, scope=layer_name + '_bottleN_' + str(0))

        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = tf.concat(layers_concat, axis=3)
            x = bottleneck_layer(x, isTrain, filters, scope=layer_name + '_bottleN_' + str(i + 1))
            layers_concat.append(x)

        x = tf.concat(layers_concat, axis=3)
        # print("concat: " , x)
        return x


def transition_layer(x, isTrain, scope):
    with tf.name_scope(scope):
        x = batch_norm(x, isTrain, name=scope + '_batch1')
        # print('TL bn: ', x)
        x = tf.nn.relu(x)
        # x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')

        # https://github.com/taki0112/Densenet-Tensorflow/issues/10

        in_channel = x.get_shape().as_list()
        in_channel = in_channel[-1] * 0.5
        x = conv2d_PC(x, in_channel, ks=1, name=scope + '_conv1')
        # print('TL conv: ', x)
        x = tf.layers.dropout(x, 0.5, isTrain)
        x = Average_pooling(x, pool_size=[2, 2], stride=2)
        # print('TL Avg Pool: ', x)
        return x


def transition_layer_breast(x, isTrain, outChannel, scope, stride=2):
    with tf.name_scope(scope):
        x = batch_norm(x, isTrain, name=scope + '_batch1')
        # print('TL bn: ', x)
        x = tf.nn.relu(x)
        # x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')

        # https://github.com/taki0112/Densenet-Tensorflow/issues/10
        x = conv2d_PC(x, outChannel, ks=1, name=scope + '_conv1')
        # print('TL conv: ', x)
        x = tf.layers.dropout(x, 0.5, isTrain)
        x = Average_pooling(x, pool_size=[2, 2], stride=stride)
        # print('TL Avg Pool: ', x)
        return x


def transition_layer_breast_last(x, isTrain, outChannel, scope):
    with tf.name_scope(scope):
        x = batch_norm(x, isTrain, name=scope + '_batch1')
        # print('TL bn: ', x)
        x = tf.nn.relu(x)
        # x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')

        # https://github.com/taki0112/Densenet-Tensorflow/issues/10
        x = conv2d_PC(x, outChannel, ks=1, name=scope + '_conv1')
        # print('TL conv: ', x)
        x = tf.layers.dropout(x, 0.5, isTrain)
        return x


def conv_2d_BN_Relu(inputae, n_filters, kernel, stride, padding, isTrain):
    conv = conv2d_PC(inputae, n_filters, ks=kernel, s=stride, padding=padding)
    bn = batch_norm(conv, isTrain, name='bn')
    return tf.nn.relu(bn), conv, bn
# End pretrained classifier.............................................................................


def pretrained_classifier(inputae, n_label, reuse, name='classifier', isTrain=False, n_filters=64, output_bias=None):
    print("Classifier", isTrain)

    if output_bias is not None:
        output_bias = tf.constant_initializer(output_bias)
    padw = 3
    with tf.variable_scope(name) as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        print("inputae: ", inputae)
        pad_input = tf.pad(inputae, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        # print("pad_input: ", pad_input)
        conv1, _conv, _bn = conv_2d_BN_Relu(pad_input, n_filters, 7, 2, 'VALID', isTrain)
        # print("conv1: ", conv1)
        padw = 1
        pad_conv1 = tf.pad(conv1, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        # print("pad_conv1: ", pad_conv1)
        pool1 = Max_Pooling(pad_conv1, pool_size=[3, 3], stride=2)
        # print("pool1: ", pool1)
        # Block 1
        block1 = dense_block(pool1, nb_layers=6, layer_name='dense_1', isTrain=isTrain, filters=n_filters)
        transition1 = transition_layer(block1, isTrain, scope='trans_1')
        # print("block1 output: ", transition1)
        # print("............................")
        block2 = dense_block(transition1, nb_layers=12, layer_name='dense_2', isTrain=isTrain, filters=n_filters)
        transition2 = transition_layer(block2, isTrain, scope='trans_2')
        # print("block2 output: ", transition2)
        # print("............................")
        block3 = dense_block(transition2, nb_layers=24, layer_name='dense_3', isTrain=isTrain, filters=n_filters)
        transition3 = transition_layer(block3, isTrain, scope='trans_3')
        # print("block3 output: ", transition3)
        # print("............................")
        block4 = dense_block(transition3, nb_layers=16, layer_name='dense_final', isTrain=isTrain, filters=n_filters)
        # print("block4 output: ", block4)
        # print("............................")
        bn = batch_norm(block4, is_training=isTrain, name='linear_batch')
        print("bn final: ", bn)
        rel = tf.nn.relu(bn)
        shape = rel.get_shape().as_list()
        # print(shape)
        gap = Average_pooling(rel, pool_size=[shape[1], shape[2]], stride=1)
        # gap = Average_pooling(rel, pool_size=[7,7], stride=1)
        # print("global avg pooling final: ", gap)
        flat = flatten(gap)
        # print("flat: ", flat)
        logit = tf.layers.dense(flat, units=n_label, bias_initializer=output_bias, name='linear')
        # print("logit: ", logit)
        if isTrain == False:
            # print(isTrain)
            logit = tf.stop_gradient(logit)
        prediction = tf.nn.sigmoid(logit)
        # pred_y = tf.argmax(prediction, 1)
        return logit, prediction