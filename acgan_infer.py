import sys, os
import tensorflow as tf
import numpy as np
import scipy.misc
import skimage.io
import skimage.transform

from ops import *
from acgan_model import *

LOG_DIR = sys.argv[1]
attr = sys.argv[2]
BATCH_SIZE = 10

# Define Network
with tf.variable_scope('input'):
    z_dim = 100
    tag_dim = 29
    z = tf.placeholder(tf.float32, [BATCH_SIZE, z_dim], name='z')
    real_tag = tf.placeholder(tf.float32, [BATCH_SIZE, tag_dim], name='real_tag')

with tf.variable_scope('generator'):
    fake_img = build_dec(z, real_tag)

# initialize and saver
init_op = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=5)
sess = tf.Session()

# if model exist, restore, else init a new one
ckpt = tf.train.get_checkpoint_state(LOG_DIR)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("=====Reading model parameters from %s=====" % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    prev_step_num = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
else:
    print("=====Model Loading Error=====")
    exit()

try:
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(1):
        if coord.should_stop():
            break
        
        # generate noise z and a batch of real images
        batch_z = np.array(np.random.multivariate_normal(np.zeros(z_dim, dtype=np.float32),
            np.identity(z_dim, dtype=np.float32), BATCH_SIZE), dtype=np.float32)
        batch_real_tags = np.reshape(
            np.tile(np.expand_dims(attr_txt2vec(attr), 0),[1,BATCH_SIZE]),
            [BATCH_SIZE, tag_dim]
                )
        print(batch_real_tags)
        
        fake_img_eval = sess.run(fake_img, feed_dict={z:batch_z, real_tag:batch_real_tags})
        
        for idx, img in enumerate(fake_img_eval):
            scipy.misc.imsave('result/%d.jpg' % idx, img)

except Exception as e:
    coord.request_stop(e)
finally :
    coord.request_stop()
    coord.join(threads)

    sess.close()
