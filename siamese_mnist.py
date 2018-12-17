"""
==================================================
Siamese implementation using Tensorflow with MNIST
==================================================
This siamese network embeds a 28x28 image (a point in 784D) into a point in 2D.
By Youngwook Paul Kwon (young at berkeley.edu)
"""
print(__doc__)

from builtins import input
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

#x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
#y_ = tf.placeholder(tf.float32, shape=[None, 10])
#
#def weight_variable(shape):
#  initial = tf.truncated_normal(shape, stddev=0.1)
#  return tf.Variable(initial)
#
#def bias_variable(shape):
#  initial = tf.constant(0.1, shape=shape)
#  return tf.Variable(initial)
#
#def conv2d(x, W):
#  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#def max_pool_2x2(x):
#  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                        strides=[1, 2, 2, 1], padding='SAME')
#
#x_image = tf.reshape(x, [-1,28,28,1])
#
#W_conv1 = weight_variable([5, 5, 1, 32])
#b_conv1 = bias_variable([32])
#
#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#h_pool1 = max_pool_2x2(h_conv1)
#
#W_conv2 = weight_variable([5, 5, 32, 64])
#b_conv2 = bias_variable([64])
#
#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#h_pool2 = max_pool_2x2(h_conv2)
#
#W_fc1 = weight_variable([7 * 7 * 64, 1024])
#b_fc1 = bias_variable([1024])
#
#h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#
#W_fc2 = weight_variable([1024, 10])
#b_fc2 = bias_variable([10])
#
#y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name="softmax")


class siamese:

  def __init__(self):
    self.x1 = tf.placeholder(tf.float32, shape=[None, 784], name="x1")
    self.x2 = tf.placeholder(tf.float32, shape=[None, 784], name="x2")

    with tf.variable_scope("siamese") as scope:
      self.o1 = self.network(self.x1)
      scope.reuse_variables()
      self.o2 = self.network(self.x2)

    self.y_ = tf.placeholder(tf.float32, [None])
    self.loss = self.loss()

  def network(self, x):
    weights = []
    fc1 = self.fc_layer(x, 1024, "fc1")
    ac1 = tf.nn.relu(fc1)
    fc2 = self.fc_layer(ac1, 1024, "fc2")
    ac2 = tf.nn.relu(fc2)
    fc3 = self.fc_layer(ac2, 2, "fc3")
    return fc3

  def fc_layer(self, bottom, n_weight, name):
    assert len(bottom.get_shape()) == 2
    n_prev_weight = bottom.get_shape()[1]
    initer = tf.truncated_normal_initializer(stddev=0.01)
    W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
    b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
    fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
    return fc

  # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
  def loss(self):
    C = tf.constant(5.0, name="C")  # margin
    eucd = tf.reduce_sum(tf.pow(tf.subtract(self.o1, self.o2), 2), 1)
    pos = tf.multiply(self.y_, eucd, name="pos_eucd")
    neg = tf.multiply(tf.subtract(1.0, self.y_), tf.pow(tf.maximum(tf.subtract(C, tf.sqrt(eucd+1e-6)), 0), 2), name="neg_eucd")
    loss = tf.reduce_mean(tf.add(pos, neg), name="loss")
    return loss


def visualize(embed, x_test, y_test, step):
  # two ways of visualization: scale to fit [0,1] scale
  # feat = embed - np.min(embed, 0)
  # feat /= np.max(feat, 0)

  # two ways of visualization: leave with original scale
  feat = embed
  ax_min = np.min(embed,0)
  ax_max = np.max(embed,0)
  ax_dist_sq = np.sum((ax_max-ax_min)**2)

  plt.figure()
  ax = plt.subplot(111)
  colormap = plt.get_cmap('tab10')
  shown_images = np.array([[1., 1.]])
  for i in range(feat.shape[0]):
    dist = np.sum((feat[i] - shown_images)**2, 1)
    if np.min(dist) < 3e-4*ax_dist_sq:   # don't show points that are too close
      continue
    shown_images = np.r_[shown_images, [feat[i]]]
    patch_to_color = np.expand_dims(x_test[i], -1)
    patch_to_color = np.tile(patch_to_color, (1, 1, 3))
    patch_to_color = (1-patch_to_color) * (1,1,1) + patch_to_color * colormap(y_test[i]/10.)[:3]
    imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(patch_to_color, zoom=0.5, cmap=plt.cm.gray_r), xy=feat[i], frameon=False)
    ax.add_artist(imagebox)

  plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
  # plt.xticks([]), plt.yticks([])
  plt.title('Embedding from the last layer of the network')
  plt.savefig("step_"+str(step)+".png")
  plt.show()

# setup siamese network
siamese = siamese();
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

# Load weights from ckpt
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('models')
if ckpt and ckpt.model_checkpoint_path:
  input_var = None
  while input_var not in ['yes', 'no']:
    input_var = input("ckpt files found. Load and continue training [yes/no]?")
  if input_var == 'yes':
    tf.logging.info('Loading checkpoint %s', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

for step in range(1001):
  batch_x1, batch_y1 = mnist.train.next_batch(128)
  batch_x2, batch_y2 = mnist.train.next_batch(128)
  batch_y = (batch_y1 == batch_y2).astype('float')

  _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
          siamese.x1: batch_x1, siamese.x2: batch_x2, siamese.y_: batch_y})

  if np.isnan(loss_v):
    print('Model diverged with loss = NaN')
    quit()

  if step % 100 == 0:
    print ('step %d: loss %.3f' % (step, loss_v))

  if step % 1000 == 0 and step > 0:
    saver.save(sess, 'models/model.ckpt')
    embed = siamese.o1.eval({siamese.x1: mnist.test.images})
    embed.tofile('embed.txt')

    # visualize result
    x_test = mnist.test.images.reshape([-1, 28, 28])
    y_test = mnist.test.labels
    visualize(embed, x_test, y_test, step)

sess.close()

