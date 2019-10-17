import sys
sys.path.append('../Utils/')
sys.path.append('../Data Pipeline/')
import tensorflow as tf
import imgaug as ia
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
import argparse
# ------------- #
import arch_ops as ops
import init_helper
import losses as ll
from montage import make_montage
from DataGetter import get_data

np.random.seed(677)
tf.set_random_seed(677)
ia.seed(678)

class AbstractGAN:

  def __init__(self, 
               data_set, 
               batch_size, 
               noise_dim, 
               loss_type,
               noise_type,
               learning_rate=0.00015,
               sess=None,
               name="gan"):
    
    self.name = name
    self.learning_rate = learning_rate
    
    # create a tensorflow session if None
    if sess == None:
      self.sess = tf.Session()
    else:
      self.sess = sess

    self.disc_updates = 2
    
    self.data_set = data_set
    self.DataSet_train, self.DataSet_test, self.img_size = get_data(data_set, False)
    self.batch_size = batch_size
    
    self.noise_type = noise_type
    self.noise_dim = noise_dim
    self.noise_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.noise_dim], name=self.name + "/generator_noise")
    self.input_placeholder = tf.placeholder(shape=[None] + self.img_size, dtype=tf.float32, name=self.name + "/discriminator_input")

    self.build_generator_discriminator()

    self.is_training = True
    self.build_loss(loss_type)
    self.build_optimizers()

    self.saver = tf.train.Saver(max_to_keep=1000)

  def sample_noise(self, shape=None):
    """Samples noise for the generator.

    Args:
      shape: the shape of the noise to be sampled.

    Returns:
      numpy array representing the sampled noise
    """
    if shape == None:
      shape = [self.batch_size, self.noise_dim]
    if self.noise_type == "uniform":
      return np.random.uniform(0.0, 1.0, shape).astype(np.float32)
    elif self.noise_type == "normal":
      return np.random.normal(0.0, 1.0, shape).astype(np.float32)
    elif self.noise_type == "truncated":
      return None
    else:
      raise ValueError("Noise type {} isn't valild.".format(str(self.noise_type)))

  def build_generator_discriminator(self):
    """Adds the generator and discriminator to the symbolic graph."""
    self.generator = self.callable_generator(self.noise_placeholder)
    self.discriminator_fake = self.callable_discriminator(self.generator)
    self.discriminator_real = self.callable_discriminator(self.input_placeholder)

  def build_loss(self, loss_type):
    """Adds the loss to the symbolic graph.
    
    Args:
      loss_type: Either "wasserstein", "KL", or "hinge".

    Returns:
      TensorFlow tensors representing the computation of the loss.
    """
    if loss_type == "wasserstein":
      self.discriminator_loss, _, _, self.generator_loss = ll.wasserstein(self.discriminator_real, self.discriminator_fake)
      self.discriminator_loss += 10 * ll.wgangp_penalty(self.callable_discriminator, self.batch_size, self.input_placeholder, self.generator)
    elif loss_type == "KL":
      self.discriminator_loss, _, _, self.generator_loss = ll.non_saturating(self.discriminator_real, self.discriminator_fake)
    elif loss_type == "hinge":
      self.discriminator_loss, _, _, self.generator_loss = ll.hinge(self.discriminator_real, self.discriminator_fake)
    else:
      raise ValueError("Loss type must be \"wasserstein\" or \"KL\" or \"hinge\".")

  def build_optimizers(self):
    """Adds the optimizer to the computational graph."""
    update_ops = [item for item in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if self.name in item.name]
    with tf.control_dependencies(update_ops):
      self.optimizer_d = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.discriminator_loss, 
        var_list=[var for var in tf.trainable_variables() if ("discriminator" in var.name and self.name in var.name)])
      self.optimizer_g = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.generator_loss, 
        var_list=[var for var in tf.trainable_variables() if ("generator" in var.name and self.name in var.name)])
  
  def show_images(self):
    """Plots a selection of random images produced by the generator. The number of samples if the batch size."""
    feed_dict = self.feed_dict_g_test()
    fake_images = self.sess.run(self.generator, feed_dict=feed_dict)
    if self.data_set == "cifar":
      imgs = [img for img in fake_images]
    else:
      imgs = [img[:,:,0] for img in fake_images]
    m = make_montage(imgs)
    plt.axis('off')
    plt.imshow(m, cmap='gray')
    plt.show()

  def feed_dict_g(self):
    """Should be implemented to return a dictionary mapping placeholders to instances for the generator during training."""
    pass

  def feed_dict_d(self):
    """Should be implemented to return a dictionary mapping placeholders to instances for the discriminator during training."""
    pass

  def feed_dict_g_test(self):
    """Should be implemented to return a dictionary mapping placeholders to instances for the generator during testing."""
    pass

  def train(self, num_steps, close=True, show_images=False, save=False):
    """Training loop for the GAN.

    Args:
      num_steps: The number of batch updates.
      close: To close the GAN's associated TF session at the end of training.
      show_images: Bool. To display images at regular intervals during training or not.
      save: Bool. To save the model at intervals or not.
    """
    bar = Bar('Training {}'.format(self.name), max=num_steps)
    init_helper.initialize_uninitialized(self.sess)
    for step in range(num_steps):
      bar.next()
      for disc_step in range(self.disc_updates):
        feed_dict = self.feed_dict_d()
        self.sess.run([self.optimizer_d], feed_dict=feed_dict)
      feed_dict = self.feed_dict_g()
      self.sess.run([self.optimizer_g], feed_dict=feed_dict)
      if step % 1000 == 0 and step != 0 and show_images:
        self.show_images()
      if save and step in [n for n in range(0, num_steps, 100)]:
        self.saver.save(self.sess, "../../../Desktop/{}_gan/my_model_{}.ckpt".format(self.data_set, str(step)))

    bar.finish()
    if close:
      self.sess.close()

  def load_weights(self, weightfile):
    self.saver.restore(self.sess, weightfile)

class SemiConditionalGAN(AbstractGAN):

  def __init__(self,
               data_set, 
               batch_size, 
               noise_dim,
               loss_type, 
               noise_type,
               learning_rate=0.00015,
               sess=None,
               name="semiditional_gan",
               input_processor=None,
               output_processor=None):
    self.input_placeholder_2 = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="hidden_gan/input_placeholder_2")
    self.input_processor = input_processor
    self.output_processor = output_processor
    super(SemiConditionalGAN, self).__init__(data_set, batch_size, noise_dim, loss_type, noise_type, learning_rate, sess, name)

  def build_generator_discriminator(self):

    # defense gan architecture
    def callable_discriminator(fake_image, real_image):
      inputs = fake_image - real_image
      output = ops.conv2d(inputs, 128, 5, 5, 2, 2, 0.02, self.name + "/discriminator/conv2d_01", False, True)
      output = tf.nn.leaky_relu(output)
      output = ops.conv2d(output, 256, 5, 5, 2, 2, 0.02, self.name + "/discriminator/conv2d_02", False, True)
      output = tf.nn.leaky_relu(output)
      output = ops.conv2d(output, 512, 5, 5, 2, 2, 0.02, self.name + "/discriminator/conv2d_03", False, True)
      output = tf.nn.leaky_relu(output)
      output = tf.contrib.layers.flatten(output)
      output = ops.linear(output, 1, self.name + "/discriminator/logits")
      return output

    # defense gan architecture
    def callable_generator(noise, real_image, is_training=True):
      output = ops.linear(noise, 7*7*4*64, self.name + "/generator/linear_01")
      output = ops.batch_norm(output, is_training, True, True, self.name + "/generator/bn_01")
      output = tf.nn.relu(output)
      output = tf.reshape(output, [-1, 7, 7, 4*64])

      output2 = ops.conv2d(real_image, 4*64, 5, 5, 2, 2, 0.02, self.name + "/generator/conv2d_01_2", False, True)
      output2 = tf.nn.leaky_relu(output2)
      output2 = ops.conv2d(output2, 4*64, 5, 5, 2, 2, 0.02, self.name + "/generator/conv2d_02_2", False, True)
      output2 = tf.nn.leaky_relu(output2)

      output = tf.concat([output, output2], axis=3)
      
      output = ops.deconv2d(output, 2*64, 5, 5, 2, 2, 0.02, self.name + "/generator/deconv2d_01", True)
      output = ops.batch_norm(output, is_training, True, True, self.name + "/generator/bn_02")
      output = tf.nn.relu(output)
      
      output = ops.deconv2d(output, 64, 5, 5, 2, 2, 0.02, self.name + "/generator/deconv2d_02", True)
      output = ops.batch_norm(output, is_training, True, True, self.name + "/generator/bn_03")
      output = tf.nn.relu(output)
      
      output = ops.deconv2d(output, 1, 5, 5, 1, 1, 0.02, self.name + "/generator/deconv2d_04", True)
      output = tf.nn.sigmoid(output)
      return output
    
    self.discriminator_real = callable_discriminator(self.input_placeholder, self.input_placeholder_2)
    self.callable_discriminator = callable_discriminator

    if self.input_processor == None:
      inputs = self.input_placeholder_2
    else:
      inputs = self.input_processor(self.input_placeholder_2)

    self.generator = callable_generator(self.noise_placeholder, inputs)
    self.discriminator_fake = callable_discriminator(self.generator, self.input_placeholder_2)
    self.callable_generator = callable_generator

  def feed_dict_train(self):
    batch, batch_pair, _ = self.DataSet_train.next_batch(self.batch_size)    
    noise_sample = self.sample_noise()
    feed_dict = {self.input_placeholder: batch, self.input_placeholder_2: batch_pair, self.noise_placeholder: noise_sample}
    return feed_dict

  def feed_dict_test(self):
    batch, batch_pair, _ = self.DataSet_test.next_batch(self.batch_size)
    noise_sample = self.sample_noise()
    feed_dict = {self.input_placeholder: batch, self.input_placeholder_2: batch_pair, self.noise_placeholder: noise_sample}
    return feed_dict

  def show_images(self):
    batch, batch_pair, _ = self.DataSet_test.next_batch(self.batch_size)
    noise_sample = self.sample_noise()
    feed_dict = {self.input_placeholder: batch, self.input_placeholder_2: batch_pair, self.noise_placeholder: noise_sample}
    fake_images = self.sess.run(self.generator, feed_dict=feed_dict)
    blanks = np.zeros([8] + self.img_size)
    images = np.vstack((batch, blanks, fake_images, blanks))
    imgs = [img[:,:,0] for img in images]
    m = make_montage(imgs)
    plt.axis('off')
    plt.imshow(m, cmap='gray')
    plt.show()

  def train(self, num_steps, close=True, show_images=False, save=False):
    bar = Bar('Training {}'.format(self.name), max=num_steps)
    init_helper.initialize_uninitialized(self.sess)
    for step in range(num_steps):
      bar.next()
      for disc_step in range(self.disc_updates):
        feed_dict = self.feed_dict_train()
        self.sess.run([self.optimizer_d], feed_dict=feed_dict)
      feed_dict = self.feed_dict_test()
      self.sess.run([self.optimizer_g], feed_dict=feed_dict)
      if step % 2500 == 0 and step != 0 and show_images:
        self.show_images()
    bar.finish()
    if close:
      self.sess.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--noise_dim", type=int, default=49, help="the dimension of the latent code z for the generator")
  parser.add_argument("--noise_type", type=str, default="normal", help="the latent noise type, one of \"normal\", \"uniform\", or \"truncated\"")
  parser.add_argument("--loss_type", type=str, default="hinge", help="the loss type, one of \"wasserstein\", \"hinge\", or \"KL\"")
  parser.add_argument("--learning_rate", type=float, default=0.00015, help="learning rate")
  parser.add_argument("--batch_size", type=int, default=64, help="batch size for training and testing")
  parser.add_argument("--num_steps", type=int, default=10001, help="number of training steps")
  parser.add_argument("--show_images", type=bool, default=True, help="bool determing if images are show every 3000 steps or not")
  parser.add_argument("--save", type=bool, default=False)
  args = parser.parse_args()
  sess = tf.Session()
  gan = SemiConditionalGAN(data_set="mnist_paired", 
                           batch_size=args.batch_size, 
                           noise_dim=args.noise_dim, 
                           loss_type=args.loss_type,
                           noise_type=args.noise_type,
                           learning_rate=args.learning_rate,
                           sess=sess,
                           name="semiditional_gan")
  gan.train(args.num_steps, close=True, show_images=args.show_images, save=args.save)