import tensorflow as tf

import numpy as np

import cPickle


# Set global parameters to the inference mode
tf.GLOBAL = {}
tf.GLOBAL['trainable'] = True
tf.GLOBAL['init'] = False
tf.GLOBAL['dropout'] = 0.0


def PyramidPixelCNN(x, seed, channels=100, D=2, L=2, l=3, num_outputs=100):

    conditioning = None
    if seed is not None:

        layer = conv(seed, "conv_first", [3, 3], channels, stride=1, padding='SAME')

        skip_connections = []
        # Down pass
        for i in range(D):
            for j in range(L):
                layer = gated_resnet(layer, "down_block%d_%d" % (i, j))
                skip_connections.append(layer)
            if i < D - 1:
                layer = conv(layer, "downsample%d" % i, [3, 3], channels, stride=2)

        # Up pass
        for i in range(D + 1):
            for j in range(L):
                layer = gated_resnet(layer, "up_block%d_%d" % (i, j), a=skip_connections.pop() if skip_connections else None)
            if i < D:
                layer = deconv(layer, "upsample%d" % i, [3, 3], channels, stride=2)

        conditioning = conv(layer, "conv_representation", [3, 3], channels, stride=1, nonlinearity=None, padding='SAME')

    # ######################################

    xs = int_shape(x)
    x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)

    u = down_shift(down_shifted_conv2d(x_pad, "conv_down", filter_size=[2, 3], out_channels=channels))
    ul = down_shift(down_shifted_conv2d(x_pad, "conv_down_2",  filter_size=[1, 3], out_channels=channels)) + \
        right_shift(down_right_shifted_conv2d(x_pad, "conv_down_right", filter_size=[2, 1], out_channels=channels))

    for rep in range(l):
        u = gated_resnet(u, "draw_down%d" % rep, a=conditioning, conv=down_shifted_conv2d)
        ul = gated_resnet(ul, "draw_down_right%d" % rep, a=tf.concat([u, conditioning], 3) if conditioning is not None else u, conv=down_right_shifted_conv2d)

    x_out = conv(tf.nn.elu(ul), "conv_last", [1, 1], num_outputs)

    return x_out, conditioning


########################################################

def quantize(x):
        return (tf.round(x * 127.5 + 127.5) - 127.5) / 127.5


def avg_pool(x, size):
    return tf.nn.avg_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')


def get_pyramid(x, N):
    return [quantize(x)] + [quantize(avg_pool(x, 2 ** (i + 1))) for i in range(N)] + [None]


def pool(inp, name, kind, size, stride, padding='SAME'):

    assert kind in ['max', 'avg']

    strides = [1, stride, stride, 1]
    sizes = [1, size, size, 1]

    with tf.variable_scope(name):
        if kind == 'max':
            out = tf.nn.max_pool(inp, sizes, strides=strides, padding=padding, name=kind)
        else:
            out = tf.nn.avg_pool(inp, sizes, strides=strides, padding=padding, name=kind)

    return out


def transform_caffe_weights(params):

    new_params = {}

    prev_s = 7

    for layer, value in params.items():
        if len(value[0].data[...].shape) == 2:
            d1 = value[0].data.shape[0]
            value_4 = np.array(value[0].data[...]).reshape([d1, -1, prev_s, prev_s])
            prev_s = 1
        else:
            value_4 = np.array(value[0].data[...])
        new_params['%s/W' % (layer, )] = value_4.transpose([2, 3, 1, 0])
        new_params['%s/b' % (layer, )] = np.array(value[1].data[...])[None, None, None, :]
    return new_params


def get_weight_initializer(params):

    initializer = []

    scope = tf.get_variable_scope()
    scope.reuse_variables()
    for layer, value in params.items():
        try:
            op = tf.get_variable('%s' % layer).assign(value)
        except:
            continue
        initializer.append(op)
    return initializer


def save_model(name, scope, sess, keys=tf.GraphKeys.TRAINABLE_VARIABLES):
    variables = tf.get_collection(keys, scope=scope)
    d = [(v.name.split(':')[0], sess.run(v)) for v in variables]

    cPickle.dump(d, open(name, 'w'), protocol=2)


# This code is a minor adaptation of the PixelCNN++ project: https://github.com/openai/pixel-cnn

def prepro(x):
    return np.cast[np.float32]((x - 127.5) / 127.5)


def unprepro(x):
    return np.cast[np.float32]((x * 127.5) + 127.5)


def concat_elu(x):
    axis = len(x.get_shape()) - 1
    return tf.nn.elu(tf.concat([x, -x], axis))


def conv(inp, name, filter_size, out_channels, stride=1, padding='SAME', nonlinearity=None, init_scale=1.0):

    with tf.variable_scope(name):

        strides = [1, stride, stride, 1]
        in_channels = inp.get_shape().as_list()[3]

        if tf.GLOBAL['init']:
            V = tf.get_variable('V', shape=filter_size + [in_channels, out_channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2])
            out = tf.nn.conv2d(inp, V_norm, strides, padding)
            m_init, v_init = tf.nn.moments(out, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', shape=None, dtype=tf.float32, initializer=scale_init, trainable=True)
            b = tf.get_variable('b', shape=None, dtype=tf.float32, initializer=-m_init * scale_init, trainable=True)
            out = tf.reshape(scale_init, [1, 1, 1, out_channels]) * (out - tf.reshape(m_init, [1, 1, 1, out_channels]))
            if nonlinearity is not None:
                out = nonlinearity(out)
            return out

        else:
            V, g, b = tf.get_variable('V'), tf.get_variable('g'), tf.get_variable('b')
            tf.assert_variables_initialized([V, g, b])

            W = g[None, None, None] * tf.nn.l2_normalize(V, [0, 1, 2])

            out = tf.nn.conv2d(inp, W, strides, padding) + b[None, None, None]

            if nonlinearity is not None:
                out = nonlinearity(out)

    return out


def deconv(inp, name, filter_size, out_channels, stride=1,
           padding='SAME', nonlinearity=None, init_scale=1.0):

    with tf.variable_scope(name):

        strides = [1, stride, stride, 1]
        [N, H, W, in_channels] = inp.get_shape().as_list()

        if padding == 'SAME':
            target_shape = [N, H * stride, W * stride, out_channels]
        else:
            target_shape = [N, H * stride + filter_size[0] - 1, W * stride + filter_size[1] - 1, out_channels]

        if tf.GLOBAL['init']:
            V = tf.get_variable('V', shape=filter_size + [out_channels, in_channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2])
            out = tf.nn.conv2d_transpose(inp, V_norm, target_shape, strides, padding)
            m_init, v_init = tf.nn.moments(out, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', shape=None, dtype=tf.float32, initializer=scale_init, trainable=True)
            b = tf.get_variable('b', shape=None, dtype=tf.float32, initializer=-m_init * scale_init, trainable=True)
            out = tf.reshape(scale_init, [1, 1, 1, out_channels]) * (out - tf.reshape(m_init, [1, 1, 1, out_channels]))
            if nonlinearity is not None:
                out = nonlinearity(out)
            return out

        else:
            V, g, b = tf.get_variable('V'), tf.get_variable('g'), tf.get_variable('b')
            tf.assert_variables_initialized([V, g, b])

            W = g[None, None, None] * tf.nn.l2_normalize(V, [0, 1, 2])

            out = tf.nn.conv2d_transpose(inp, W, target_shape, strides, padding) + b[None, None, None]

            if nonlinearity is not None:
                out = nonlinearity(out)

    return out


def gated_resnet(x, name, nonlinearity=concat_elu, conv=conv, a=None):
    with tf.variable_scope(name):
        num_filters = int(x.get_shape()[-1])

        c1 = conv(nonlinearity(x), "conv1", [3, 3], num_filters)

        if a is not None:
            c1 += conv(nonlinearity(a), "conv_aux", [1, 1], num_filters)

        c1 = nonlinearity(c1)

        if tf.GLOBAL['dropout'] > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1.0 - tf.GLOBAL['dropout'])

        c2 = conv(c1, "conv2", [3, 3], num_filters * 2, init_scale=0.1)

        a, b = tf.split(c2, 2, 3)
        c3 = a * tf.nn.sigmoid(b)
        return x + c3


def down_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0], 1, xs[2], xs[3]]), x[:, :xs[1] - 1, :, :]], 1)


def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0], xs[1], 1, xs[3]]), x[:, :, :xs[2] - 1, :]], 2)


def down_shifted_conv2d(x, name, filter_size, out_channels, stride=1, nonlinearity=None, init_scale=1.0):
    x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0], [int((filter_size[1] - 1) / 2), int((filter_size[1] - 1) / 2)], [0, 0]])
    return conv(x, name, filter_size, out_channels, stride=stride, padding='VALID', nonlinearity=nonlinearity, init_scale=init_scale)


def down_right_shifted_conv2d(x, name, filter_size, out_channels, stride=1, nonlinearity=None, init_scale=1.0):
    x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0], [filter_size[1] - 1, 0], [0, 0]])
    return conv(x, name, filter_size, out_channels, stride=stride, padding='VALID', nonlinearity=nonlinearity, init_scale=init_scale)


def down_shifted_deconv2d(x, name, filter_size, out_channels, stride=1, nonlinearity=None, init_scale=1.0):
    x = deconv(x, name, channels=out_channels, filter_size=filter_size, pad='VALID', stride=[stride, stride], init_scale=init_scale)
    xs = int_shape(x)
    return x[:, :(xs[1] - filter_size[0] + 1), int((filter_size[1] - 1) / 2):(xs[2] - int((filter_size[1] - 1) / 2)), :]


def down_right_shifted_deconv2d(x, name, filter_size, out_channels, stride=1, nonlinearity=None, init_scale=1.0):
    x = deconv(x, name, out_channels=out_channels, filter_size=filter_size, pad='VALID', stride=[stride, stride], init_scale=init_scale)
    xs = int_shape(x)
    return x[:, :(xs[1] - filter_size[0] + 1):, :(xs[2] - filter_size[1] + 1), :]


def int_shape(x):
    return list(map(int, x.get_shape()))


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keep_dims=True))


def sample_from_discretized_mix_logistic(l, scale_var=0.0, nr_mix=10):
    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:, :, :, :, :nr_mix] * sel, 4)
    log_scales = tf.maximum(tf.reduce_sum((scale_var + l[:, :, :, :, nr_mix:2 * nr_mix]) * sel, 4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, 4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:, :, :, 0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    return tf.concat([tf.reshape(x0, xs[:-1] + [1]), tf.reshape(x1, xs[:-1] + [1]), tf.reshape(x2, xs[:-1] + [1])], 3)
