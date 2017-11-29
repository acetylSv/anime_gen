from ops import *

def tag_transform(tag):
    tag_h_dim = 32
    tag_shape = tag.get_shape().as_list()
    
    with tf.variable_scope('tag_transform'):
        tag_h = linear(tag, tag_h_dim, name='tag_linear')
        tag_h = tf.nn.relu(tag_h)

    return tag_h

def build_dec(source, tag_h):
    source_shape = source.get_shape().as_list()
    tag_shape = tag_h.get_shape().as_list()
    batch_size = source_shape[0]
    h0_shape = [batch_size, 4, 4, 1024]
    h1_shape = [batch_size, 8, 8, 512]
    h2_shape = [batch_size, 16, 16, 256]
    h3_shape = [batch_size, 32, 32, 128]
    output_shape = [batch_size, 64, 64, 3]

    with tf.variable_scope('project_and_reshape'):
        source_with_condition = tf.concat([source, tag_h], axis=1)
        lin_dim = np.prod(np.array(h0_shape[1:]))
        hidden = linear(source_with_condition, lin_dim, name='dec_project_linear')
        h0 = tf.reshape(hidden, h0_shape)
        #h0 = batch_norm(h0, name='lt_bn')
        h0 = instance_norm(h0, name='lt_bn')
        h0 = tf.nn.relu(h0)

    with tf.variable_scope('deconv_1'):
        h1 = deconv2d(h0, h1_shape, name='dec_deconv2d_1')
        #h1 = batch_norm(h1, name='dec_deconv2d_bn_1')
        h1 = instance_norm(h1, name='dec_deconv2d_bn_1')
        h1 = tf.nn.relu(h1)

    with tf.variable_scope('deconv_2'):
        h2 = deconv2d(h1, h2_shape, name='dec_deconv2d_2')
        #h2 = batch_norm(h2, name='dec_deconv2d_bn_2')
        h2 = instance_norm(h2, name='dec_deconv2d_bn_2')
        h2 = tf.nn.relu(h2)

    with tf.variable_scope('deconv_3'):
        h3 = deconv2d(h2, h3_shape, name='dec_deconv2d_3')
        #h3 = batch_norm(h3, name='dec_deconv2d_bn_3')
        h3 = instance_norm(h3, name='dec_deconv2d_bn_3')
        h3 = tf.nn.relu(h3)

    with tf.variable_scope('dec_output'):
        output = deconv2d(h3, output_shape, name='dec_deconv2d_4')
        # conv with strides=1
        output = conv2d(output, output_shape[-1], d_h=1, d_w=1, name='dec_conv2d')
        output = tf.nn.tanh(output)/2.0 + 0.5 # normalize

    print(h0.get_shape())
    print(h1.get_shape())
    print(h2.get_shape())
    print(h3.get_shape())
    print(output.get_shape())

    return output

def build_critic(source, tag_h):
    # no BN in discriminator
    source_shape = source.get_shape().as_list()
    tag_shape = tag_h.get_shape().as_list()
    if source_shape[1] is None:
        print('Source reshaping')
        source = tf.reshape(source, [-1, 64, 64, 3])
        source_shape = source.get_shape().as_list()
        print(source_shape)

    with tf.variable_scope('conv_1'):
        h1 = conv2d(source, source_shape[1], name='dis_conv2d_1')
        #h1 = batch_norm(h1, name='dis_conv2d_bn_1')
        h1 = instance_norm(h1, name='dis_conv2d_in_1')
        h1 = lrelu(h1)
    with tf.variable_scope('conv_2'):
        h2 = conv2d(h1, source_shape[1]*2, name='dis_conv2d_2')
        #h2 = batch_norm(h2, name='dis_conv2d_bn_2')
        h2 = instance_norm(h2, name='dis_conv2d_in_2')
        h2 = lrelu(h2)
    with tf.variable_scope('conv_3'):
        h3 = conv2d(h2, source_shape[1]*4, name='dis_conv2d_3')
        #h3 = batch_norm(h3, name='dis_conv2d_bn_3')
        h3 = instance_norm(h3, name='dis_conv2d_in_3')
        h3 = lrelu(h3)
    with tf.variable_scope('conv_4'):
        h4 = conv2d(h3, source_shape[1]*8, name='dis_conv2d_4')
        #h4 = batch_norm(h4, name='dis_conv2d_bn_4')
        h4 = instance_norm(h4, name='dis_conv2d_in_4')
        h4 = lrelu(h4)
    with tf.variable_scope('concat_tag'):
        current_dim = h4.get_shape().as_list()
        tag_h = tf.expand_dims(tag_h, 1)
        tag_h = tf.expand_dims(tag_h, 1)
        target = tf.tile(tag_h, [1, current_dim[1], current_dim[1], 1])
        h_concat = tf.concat([h4, target], axis=3)
    with tf.variable_scope('one_one_conv'):
        h5 = conv2d(h_concat, h_concat.get_shape().as_list()[-1]//2, k_h=1, k_w=1,
            d_h=1, d_w=1, name='dis_1x1_conv2d')
        #h5 = batch_norm(h5, name='dis_1x1_conv2d_bn')
        h5 = instance_norm(h5, name='dis_1x1_conv2d_in')
        h5 = lrelu(h5)
    with tf.variable_scope('output'):
        output = conv2d(h5, 1, k_h=4, k_w=4, d_h=1, d_w=1,
            name='dis_output_conv2d', padding='VALID')

    print(source.get_shape())
    print(h1.get_shape())
    print(h2.get_shape())
    print(h3.get_shape())
    print(h4.get_shape())
    print(h_concat.get_shape())
    print(h5.get_shape())
    print(output.get_shape())

    return tf.nn.sigmoid(output), output
