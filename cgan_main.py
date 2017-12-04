import numpy as np
import random
from ops import *
from cgan_model import *
import tensorflow as tf

#tf.set_random_seed(123)
#np.random.seed(123)
#random.seed(123)
def _shuffle(X):
    randomize = np.arange(len(X), dtype=np.int32)
    np.random.shuffle(randomize)
    return(np.array(X)[randomize])

TARGET = 'condition_anime_gen'
LOG_DIR = './log/'+TARGET
DATA_DIR = './data/faces'

LEARNING_RATE = 0.0002
BETA_1 = 0.5
BETA_2 = 0.9

LAMBDA = 10

BATCH_SIZE = 64

MAX_ITERATION = 150000
SAVE_PERIOD = 1000
SUMMARY_PERIOD = 200

NUM_CRITIC_TRAIN = 6
NUM_GEN_TRAIN = 1

# Load Data
from glob import glob
import os
import scipy.misc
import skimage.io
import skimage.transform
input_fname_pattern = '*.jpg'
data = glob(os.path.join(DATA_DIR, input_fname_pattern))
file_name_idx = []
for i in data:
    file_name_idx.append(i.strip().split('/')[-1].split('.jpg')[0])
img_list = [skimage.transform.resize(scipy.misc.imread(x), (64, 64)) for x in data]
tagvec = np.load(open('tag2vec/tag_vec.npy', 'rb'))[np.array(file_name_idx, dtype=np.int32)]
rdm_tagvec = _shuffle(np.array(tagvec))
#img_list = [skimage.transform.resize(scipy.misc.imread(x), (64, 64)) for idx, x in enumerate(data) if idx < 500]

# Define Network
with tf.variable_scope('input'):
    z_dim = 100
    tag_dim = 29
    z = tf.placeholder(tf.float32, [BATCH_SIZE, z_dim], name='z')
    real_tag = tf.placeholder(tf.float32, [BATCH_SIZE, tag_dim], name='real_tag')
    fake_tag = tf.placeholder(tf.float32, [BATCH_SIZE, tag_dim], name='fake_tag')
    real_img = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3], name='real_img')

with tf.variable_scope('generator_tag_h'):
    real_tag_h_gen = tag_transform(real_tag)

with tf.variable_scope('generator'):
    fake_img = build_dec(z, real_tag_h_gen)

with tf.variable_scope('interpolate'):
    alpha = tf.random_uniform(shape=[BATCH_SIZE,1,1,1], minval=0.,maxval=1.)
    interpolates = alpha * real_img + (1 - alpha) * fake_img

with tf.variable_scope('discriminator_tag_h') as scope:
    real_tag_h_dis = tag_transform(real_tag)
    scope.reuse_variables()
    fake_tag_h_dis = tag_transform(fake_tag)

with tf.variable_scope('discriminator') as scope:
    _, v_r = build_critic(real_img, real_tag_h_dis)
    scope.reuse_variables()
    _, v_w = build_critic(real_img, fake_tag_h_dis)
    _, v_f = build_critic(fake_img, real_tag_h_dis)

    #_, v_hat_w = build_critic(interpolates, fake_tag_h)
    _, v_hat_f = build_critic(interpolates, real_tag_h_dis)

c_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]

# show variables
#for v in c_vars : print(v)
#print('----------------------')
#for v in g_vars : print(v)

# Define Loss and Optimizer
c_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, BETA_1, BETA_2)
g_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, BETA_1, BETA_2)

# Discriminator Loss
W = tf.reduce_mean(v_r) - ((tf.reduce_mean(v_w) + tf.reduce_mean(v_f)) * 0.5)
#GP = tf.reduce_mean(
#        (tf.sqrt(tf.reduce_sum(tf.gradients(v_fake, fake_img)[0]**2,reduction_indices=[1,2,3]))-1.0)**2
#     )
#GP1 = tf.reduce_mean(
#        (tf.sqrt(tf.reduce_sum(
#            tf.gradients(v_hat_w, interpolates)[0]**2,reduction_indices=[1,2,3]))-1.0)**2
#     )
GP2 = tf.reduce_mean(
        (tf.sqrt(tf.reduce_sum(
            tf.gradients(v_hat_f, interpolates)[0]**2,reduction_indices=[1,2,3]))-1.0)**2
     )
#GP = GP1+GP2
GP = GP2

loss_c = -1.0*W + LAMBDA*GP
with tf.variable_scope('c_train'):
    gvs = c_optimizer.compute_gradients(loss_c, var_list=c_vars)
    train_c_op = c_optimizer.apply_gradients(gvs)

# Generator Loss
loss_g = -1.0 * tf.reduce_mean(v_f)
with tf.variable_scope('g_train'):
    gvs = g_optimizer.compute_gradients(loss_g, var_list=g_vars)
    train_g_op  = g_optimizer.apply_gradients(gvs)

# tensorboard usage
tf.summary.image('real_a', real_img, max_outputs=20)
tf.summary.image('fake_a', fake_img, max_outputs=20)
tf.summary.scalar('Estimated W', W)
tf.summary.scalar('gradient_penalty', GP)
tf.summary.scalar('loss_g', loss_g)
summary_op = tf.summary.merge_all()

# initialize and saver
init_op = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=20)
sess = tf.Session()

# if model exist, restore, else init a new one
ckpt = tf.train.get_checkpoint_state(LOG_DIR)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("=====Reading model parameters from %s=====" % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    prev_step_num = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
else:
    print("=====Init a new model=====")
    sess.run([init_op])
    prev_step_num = 0

try:
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    batch_step_num = len(img_list) // BATCH_SIZE
    for step in range(1, MAX_ITERATION+1):
        if coord.should_stop():
            break

        # shuffle real images
        if step % batch_step_num == 0:
            rdm_tagvec = _shuffle(rdm_tagvec)
            print('shuffle rdm_tagvec done')
        
        # generate noise z and a batch of real images
        batch_z = np.array(np.random.multivariate_normal(np.zeros(z_dim, dtype=np.float32),
            np.identity(z_dim, dtype=np.float32), BATCH_SIZE), dtype=np.float32)
        batch_images = np.array(img_list[(step%batch_step_num)*BATCH_SIZE:(step%batch_step_num+1)*BATCH_SIZE],
            dtype=np.float32)
        batch_real_tags = tagvec[(step%batch_step_num)*BATCH_SIZE:(step%batch_step_num+1)*BATCH_SIZE]
        batch_fake_tags = rdm_tagvec[(step%batch_step_num)*BATCH_SIZE:(step%batch_step_num+1)*BATCH_SIZE]
        
        # training discriminator
        for _ in range(NUM_CRITIC_TRAIN):
            _  = sess.run(train_c_op,
                            feed_dict={
                                real_img:batch_images,
                                z:batch_z,
                                real_tag:batch_real_tags,
                                fake_tag:batch_fake_tags
                            })
        # training generator
        for _ in range(NUM_GEN_TRAIN):
            W_eval, GP_eval, loss_g_eval, _ = sess.run([W, GP, loss_g, train_g_op],
                            feed_dict={
                                real_img:batch_images,
                                z:batch_z,
                                real_tag:batch_real_tags,
                                fake_tag:batch_fake_tags
                            })

        print('%7d : W : %1.6f, GP : %1.6f, Loss G : %1.6f' % (step, W_eval, GP_eval, loss_g_eval))
        
        if( step % SUMMARY_PERIOD == 0 ):
            print('=========================')
            print('Step %d, True Tag:' % step)
            for i in batch_real_tags:
                print(i)
                print(attr_lookup(i))
            print('=========================')
            summary_str = sess.run(summary_op,
                feed_dict={
                    real_img:batch_images,
                    z:batch_z,
                    real_tag:batch_real_tags,
                    fake_tag:batch_fake_tags
                })
            summary_writer.add_summary(summary_str, step)
        
        if( step % SAVE_PERIOD == 0 ):
            saver.save(sess, LOG_DIR+'/model.ckpt', global_step=step)

except Exception as e:
    coord.request_stop(e)
finally :
    coord.request_stop()
    coord.join(threads)

    sess.close()
