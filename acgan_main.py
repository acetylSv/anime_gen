import numpy as np
import random
from ops import *
from acgan_model import *
import tensorflow as tf

#tf.set_random_seed(123)
#np.random.seed(123)
#random.seed(123)
def _shuffle(X):
    randomize = np.arange(len(X), dtype=np.int32)
    np.random.shuffle(randomize)
    return(np.array(X)[randomize])

TARGET = 'ACGAN_anime_gen'
LOG_DIR = './log/'+TARGET
DATA_DIR = './data/faces'

LEARNING_RATE = 0.0001
BETA_1 = 0.5
BETA_2 = 0.9

BATCH_SIZE = 64
GP_lambda = 10

MAX_ITERATION = 150000
SAVE_PERIOD = 1000
SUMMARY_PERIOD = 200

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
#img_list = [skimage.transform.resize(scipy.misc.imread(x), (64, 64)) for idx, x in enumerate(data) if idx < 500]

# Define Network (Forward)
with tf.variable_scope('input'):
    z_dim = 100
    tag_dim = 29
    z = tf.placeholder(tf.float32, [BATCH_SIZE, z_dim], name='z')
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, tag_dim], name='labels')
    real_img = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3], name='real_img')
    decay_lambda = tf.placeholder(tf.float32, [], name='decay_lambda')

with tf.variable_scope('generator'):
    fake_img = build_dec(z, labels)

with tf.variable_scope('interpolate'):
    alpha = tf.random_uniform(shape=[BATCH_SIZE,1,1,1], minval=0.,maxval=1.)
    interpolates = alpha * real_img + (1 - alpha) * fake_img

with tf.variable_scope('discriminator') as scope:
    D_real, D_real_logits, hidden_real = build_critic(real_img)
    scope.reuse_variables()
    D_fake, D_fake_logits, hidden_fake = build_critic(fake_img)
    _, D_inter_logits, _ = build_critic(interpolates)

with tf.variable_scope('classifier') as scope:
    code_real, code_logit_real = build_classifier(hidden_real)
    scope.reuse_variables()
    code_fake, code_logit_fake = build_classifier(hidden_fake)

# Discriminator Loss
#   1. DCGAN D Loss
d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_real_logits, labels=tf.ones_like(D_real)
                )
            )
d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_fake_logits, labels=tf.zeros_like(D_fake)
                )
            )
#   2. Improved WGAN D loss
W = tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_fake_logits)
GP = tf.reduce_mean(
        (tf.sqrt(tf.reduce_sum(
            tf.gradients(D_inter_logits, interpolates)[0]**2,reduction_indices=[1,2,3]))-1.0)**2
    )
#   3. Total D loss
d_loss = ((1-decay_lambda)*(d_loss_real + d_loss_fake)) + \
              decay_lambda*(-1.0*W + GP_lambda*GP)

# Generator Loss
#   1. DCGAN G Loss
dc_g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_fake_logits, labels=tf.ones_like(D_fake)
                )
            )
#   2. Improved WGAN G loss
iw_g_loss = -1.0 * tf.reduce_mean(D_fake_logits)
#   3. Total G loss
g_loss = (1-decay_lambda)*dc_g_loss + \
             decay_lambda*iw_g_loss

# Classifier Loss
q_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=code_logit_real, labels=labels
                )
            )
q_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=code_logit_fake, labels=labels
                )
            )
q_loss = q_fake_loss + q_real_loss

# Define Optimizer
d_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, BETA_1, BETA_2)
g_optimizer = tf.train.AdamOptimizer(LEARNING_RATE * 2, BETA_1, BETA_2)
q_optimizer = tf.train.AdamOptimizer(LEARNING_RATE * 2, BETA_1, BETA_2)

d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
q_vars = [v for v in tf.trainable_variables() if v.name.startswith('classifier')
    or v.name.startswith('generator') or v.name.startswith('discriminator')]

# show variables
for v in d_vars : print(v)
print('----------------------')
for v in g_vars : print(v)
print('----------------------')
for v in q_vars : print(v)

with tf.variable_scope('d_train'):
    gvs = d_optimizer.compute_gradients(d_loss, var_list=d_vars)
    train_d_op = d_optimizer.apply_gradients(gvs)

with tf.variable_scope('g_train'):
    gvs = g_optimizer.compute_gradients(g_loss, var_list=g_vars)
    train_g_op  = g_optimizer.apply_gradients(gvs)

with tf.variable_scope('q_train'):
    gvs = q_optimizer.compute_gradients(q_loss, var_list=q_vars)
    train_q_op  = q_optimizer.apply_gradients(gvs)

# tensorboard usage
tf.summary.image('real_a', real_img, max_outputs=20)
tf.summary.image('fake_a', fake_img, max_outputs=20)
d_loss_sum = tf.summary.scalar("d_loss", d_loss)
d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
W_sum = tf.summary.scalar("W", W)
GP_sum = tf.summary.scalar("GP", GP)
dc_g_loss_sum = tf.summary.scalar("dc_g_loss", dc_g_loss)
iw_g_loss_sum = tf.summary.scalar("iw_g_loss", iw_g_loss)
q_loss_sum = tf.summary.scalar("q_loss", q_loss)
q_loss_real_sum = tf.summary.scalar("q_real_loss", q_real_loss)
q_loss_fake_sum = tf.summary.scalar("q_fake_loss", q_fake_loss)
decay_lambda_sum = tf.summary.scalar("decay_lambda", decay_lambda)

#g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
#d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
#q_sum = tf.summary.merge([q_loss_sum, q_real_sum, q_fake_sum])
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

    epoch = 0
    batch_step_num = len(img_list) // BATCH_SIZE
    for step in range(1, MAX_ITERATION+1):
        if coord.should_stop():
            break

        if (step % batch_step_num) == 0:
            epoch += 1

        # generate noise z and a batch of real images
        batch_z = np.array(np.random.multivariate_normal(np.zeros(z_dim, dtype=np.float32),
            np.identity(z_dim, dtype=np.float32), BATCH_SIZE), dtype=np.float32)
        batch_images = np.array(img_list[(step%batch_step_num)*BATCH_SIZE:(step%batch_step_num+1)*BATCH_SIZE],
            dtype=np.float32)
        batch_real_tags = tagvec[(step%batch_step_num)*BATCH_SIZE:(step%batch_step_num+1)*BATCH_SIZE]
        decay_lambda_rate = np.array(np.max([(1.0-epoch*0.002), 0.25]))
        
        # training discriminator
        for _ in range(5):
            _, d_loss_eval = sess.run(
                            [train_d_op, d_loss],
                            feed_dict={
                                real_img:batch_images,
                                z:batch_z,
                                labels:batch_real_tags,
                                decay_lambda:decay_lambda_rate
                            }
                        )
        # training generator and classifier
        for _ in range(1):
            _, _, g_loss_eval, q_loss_eval = sess.run(
                            [train_g_op, train_q_op, g_loss, q_loss],
                            feed_dict={
                                real_img:batch_images,
                                z:batch_z,
                                labels:batch_real_tags,
                                decay_lambda:decay_lambda_rate
                            }
                        )

        print('%7d : D_loss : %1.6f, G_loss : %1.6f, Q_loss : %1.6f' %
            (step, d_loss_eval, g_loss_eval, q_loss_eval))
        
        # summarize or not
        if( step % SUMMARY_PERIOD == 0 ):
            print('=========================')
            print('Step %d, True Tag:' % step)
            for i in batch_real_tags[:20]:
                print(i)
                print(attr_lookup(i))
            print('=========================')
            summary_str = sess.run(
                summary_op,
                feed_dict={
                    real_img:batch_images,
                    z:batch_z,
                    labels:batch_real_tags,
                    decay_lambda:decay_lambda_rate
                })
            summary_writer.add_summary(summary_str, step)
        
        # save or not
        if( step % SAVE_PERIOD == 0 ):
            saver.save(sess, LOG_DIR+'/model.ckpt', global_step=step)

except Exception as e:
    coord.request_stop(e)
finally :
    coord.request_stop()
    coord.join(threads)

    sess.close()
