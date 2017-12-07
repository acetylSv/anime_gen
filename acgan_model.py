from ops import *

def build_dec(source, labels):
    source_shape = source.get_shape().as_list()
    labels_shape = labels.get_shape().as_list()
    batch_size = source_shape[0]
    h3_shape = [batch_size, 4, 4, 1024]
    h4_shape = [batch_size, 8, 8, 512]
    h5_shape = [batch_size, 16, 16, 256]
    h6_shape = [batch_size, 32, 32, 128]
    output_shape = [batch_size, 64, 64, 3]

    with tf.variable_scope('fc_1'):
        source_with_condition = tf.concat([source, labels], axis=1)
        h1 = linear(source_with_condition, 1024, name='dec_fc_1')
        h1 = batch_norm(h1, name='dec_fc_1_bn1')
        h1 = tf.nn.relu(h1)

    with tf.variable_scope('fc_2'):
        lin_dim = np.prod(np.array(h3_shape[1:]))
        h2 = linear(h1, lin_dim, name='dec_fc_2')
        h2 = batch_norm(h2, name='dec_fc_2_bn2')
        h2 = tf.nn.relu(h2)
    
    # reshape
    h3 = tf.reshape(h2, h3_shape)

    with tf.variable_scope('deconv2d_3'):
        h4 = deconv2d(h3, h4_shape, name='dec_deconv2d_3')
        h4 = batch_norm(h4, name='dec_deconv2d_bn_3')
        h4 = tf.nn.relu(h4)
    
    with tf.variable_scope('deconv2d_4'):
        h5 = deconv2d(h4, h5_shape, name='dec_deconv2d_4')
        h5 = batch_norm(h5, name='dec_deconv2d_bn_4')
        h5 = tf.nn.relu(h5)
    
    with tf.variable_scope('deconv2d_5'):
        h6 = deconv2d(h5, h6_shape, name='dec_deconv2d_5')
        h6 = batch_norm(h6, name='dec_deconv2d_bn_5')
        h6 = tf.nn.relu(h6)

    with tf.variable_scope('dec_output'):
        output = deconv2d(h6, output_shape, name='dec_deconv2d_6')
        output = tf.nn.sigmoid(output)

    print('Generator Hidden Shape:')
    print(h1.get_shape())
    print(h2.get_shape())
    print(h3.get_shape())
    print(h4.get_shape())
    print(h5.get_shape())
    print(h6.get_shape())
    print(output.get_shape())
    print('==========')

    return output

def build_critic(source):
    source_shape = source.get_shape().as_list()

    with tf.variable_scope('conv_1'):
        h1 = conv2d(source, source_shape[1], k_h=4, k_w=4, d_h=2, d_w=2, name='dis_conv2d_1')
        # no BN here ?
        h1 = lrelu(h1)
    with tf.variable_scope('conv_2'):
        h2 = conv2d(h1, source_shape[1]*2, k_h=4, k_w=4, d_h=2, d_w=2, name='dis_conv2d_2')
        #h2 = batch_norm(h2, name='dis_conv2d_bn_2')
        h2 = lrelu(h2)
    with tf.variable_scope('conv_3'):
        h3 = conv2d(h2, source_shape[1]*4, k_h=4, k_w=4, d_h=2, d_w=2, name='dis_conv2d_3')
        #h3 = batch_norm(h3, name='dis_conv2d_bn_3')
        h3 = lrelu(h3)
    with tf.variable_scope('conv_4'):
        h4 = conv2d(h3, source_shape[1]*8, k_h=4, k_w=4, d_h=2, d_w=2, name='dis_conv2d_4')
        #h4 = batch_norm(h4, name='dis_conv2d_bn_4')
        h4 = lrelu(h4)
    with tf.variable_scope('fc_5'):
        h5 = linear(tf.contrib.layers.flatten(h4), 1024, name='dis_fc5')
        #h5 = batch_norm(h5, name='dis_fc5_bn_5')
        h5 = lrelu(h5)
    with tf.variable_scope('output'):
        out_logit = linear(h5, 1, name='dis_fc6')
        out = tf.nn.sigmoid(out_logit)

    print('Discriminator Hidden Shape:')
    print(source.get_shape())
    print(h1.get_shape())
    print(h2.get_shape())
    print(h3.get_shape())
    print(h4.get_shape())
    print(h5.get_shape())
    print(out_logit.get_shape())
    print(out.get_shape())
    print('==========')

    return out, out_logit, h5

def build_classifier(source):
    source_shape = source.get_shape().as_list()
    
    with tf.variable_scope('fc_1'):
        h1 = linear(source, 128, name='cla_fc1')
        h1 = batch_norm(h1, name='cla_fc1_bn_1')
        h1 = lrelu(h1)

    with tf.variable_scope('fc_2'):
        out_logit = linear(h1, 29, name='cla_fc2')
        # on a multi-label classification setting
        out = tf.nn.sigmoid(out_logit)
    
    print('Classifier Hidden Shape:')
    print(source.get_shape())
    print(h1.get_shape())
    print(out_logit.get_shape())
    print(out.get_shape())
    print('==========')
    
    return out, out_logit
