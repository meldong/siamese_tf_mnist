"""
==================================================
Siamese implementation using Tensorflow with MNIST
==================================================
This siamese network embeds a 28x28 image (a point in 784D) into a point in 2D.
By Youngwook Paul Kwon (young at berkeley.edu)
"""
print(__doc__)

from matplotlib import offsetbox
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)


class siamese:

  def __init__(self):
    self.x1 = tf.placeholder(tf.float32, shape=[None, 784], name="x1")
    self.x2 = tf.placeholder(tf.float32, shape=[None, 784], name="x2")

    with tf.variable_scope("siamese") as scope:
      self.o1 = self.network(self.x1)
      scope.reuse_variables()
      self.o2 = self.network(self.x2)

    self.y_ = tf.placeholder(tf.float32, shape=[None], name="y")
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

  # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||)^2
  def loss(self):
    eucd = tf.reduce_sum(tf.pow(tf.subtract(self.o1, self.o2), 2), 1)
    pos = tf.multiply(self.y_, eucd, name="pos_eucd")
    C = tf.constant(5.0, name="C")  # margin 5.0
    emax = tf.pow(tf.maximum(tf.subtract(C, tf.sqrt(eucd+1e-6)), 0.0), 2)
    neg = tf.multiply(tf.subtract(1.0, self.y_), emax, name="neg_emax")
    loss = tf.reduce_mean(tf.add(pos, neg), name="loss")
    return loss


def visualize(embed, x_test, y_test):
  feat = embed
#  feat = embed - np.min(embed, 0)
#  feat /= np.max(feat, 0)
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
    imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(patch_to_color, zoom=0.5, cmap=plt.cm.gray_r),
            xy=feat[i], frameon=False)
    ax.add_artist(imagebox)

  plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
  plt.title('Embedding from the last layer of the network')
  plt.show()


siamese = siamese();
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('models')
if ckpt and ckpt.model_checkpoint_path:
  input_var = None
  while input_var not in ['yes', 'no']:
    input_var = input("CKPT files found. Load to continue training [yes/no]?")
  if input_var == 'yes':
    saver.restore(sess, ckpt.model_checkpoint_path)

for step in range(5000):
  batch_x1, batch_y1 = mnist.train.next_batch(50)
  batch_x2, batch_y2 = mnist.train.next_batch(50)
  batch_y = (batch_y1 == batch_y2).astype('float')

  _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
          siamese.x1: batch_x1, siamese.x2: batch_x2, siamese.y_: batch_y})

  if np.isnan(loss_v):
    print('Model diverged with loss = NaN')
    quit()

  if step % 500 == 0:
    print ('step %d: loss %.3f' % (step, loss_v))

  if step % 1000 == 0 and step > 0:
    saver.save(sess, 'models/model.ckpt')
    embed = siamese.o1.eval({siamese.x1: mnist.test.images})
    embed.tofile('embed.txt')

# visualize result
x_test = mnist.test.images.reshape([-1, 28, 28])
y_test = mnist.test.labels
visualize(embed, x_test, y_test)

sess.close()

