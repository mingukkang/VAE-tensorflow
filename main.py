import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data
import VAE_model
import plot

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('n_hidden_encoder', 500, "dimension of  hidden layer for encoder")
flags.DEFINE_integer('n_z', 2, "number of sampling")
flags.DEFINE_integer('n_hidden_decoder', 500, "dimension of hidden layer for decoder")
flags.DEFINE_integer('decay_step', 2000, "decay step for learning rate decay")
flags.DEFINE_integer('num_epoch', 200, "number of epochs for training")
flags.DEFINE_integer('batch_size', 128, "number of batch_size")
flags.DEFINE_float('decay_rate', 0.96, "learning rate decay rate")
flags.DEFINE_float('learning_rate', 1e-4, "Learning rate")
flags.DEFINE_float('keep_prob', 0.5, "Dropout rate")

train_data, train_labels,validation_data,validation_labels,test_data,test_labels = data.prepare_MNIST_Data()

def train():
    X = tf.placeholder(tf.float32, shape = [None,784], name = "input_data")
    keep_prob = tf.placeholder(tf.float32, name ="dropout_rate")
    global_step = tf.Variable(0, trainable = False)

    z,X_out, Recon_error, Regularization_error, ELBO = VAE_model.Variational_autoencoder(X,
                                                                                    FLAGS.n_hidden_encoder,
                                                                                    FLAGS.n_z,
                                                                                    FLAGS.n_hidden_decoder,
                                                                                    keep_prob)

    learning_rate_decayed = FLAGS.learning_rate*FLAGS.decay_rate**(global_step/FLAGS.decay_step)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate_decayed).minimize(ELBO, global_step = global_step)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    for i in range(FLAGS.num_epoch):
        Recon_e = 0
        Regul_e = 0
        ELBO_Value = 0
        total_batch = int(len(train_data)/FLAGS.batch_size)
        for j in range(total_batch):
            batch_xs = train_data[ j*FLAGS.batch_size: j*FLAGS.batch_size + FLAGS.batch_size]
            Output, e_1,e_2, e_3,_ = sess.run([X_out, Recon_error, Regularization_error, ELBO,optimizer], feed_dict = {X:batch_xs, keep_prob:FLAGS.keep_prob})
            Recon_e += e_1/total_batch
            Regul_e += e_2/ total_batch
            ELBO_Value += e_3/total_batch
        print("Epoch: ",i, "  Reconstruction_Error: ", Recon_e, "  Regularization_Error: ",Regul_e, "  ELBO: ", ELBO_Value)

        if i %20 ==0:
            location = "./parameter" + str(i) + ".ckpt"
            saving_data = saver.save(sess,location)
    location = "./parameter" + str(i) + ".ckpt"
    saving_data = saver.save(sess,location)
    sess.close()

def test():
    X = tf.placeholder(tf.float32, shape = [None,784], name ="Input_data")
    keep_prob = tf.placeholder(tf.float32, name ="dropout_rate")
    z,X_out, Recon_error, Regularization_error, ELBO = VAE_model.Variational_autoencoder(X,
                                                                                                       FLAGS.n_hidden_encoder,
                                                                                                       FLAGS.n_z,
                                                                                                       FLAGS.n_hidden_decoder,
                                                                                                       keep_prob)
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess,"parameter" + str(FLAGS.num_epoch -1) + ".ckpt")

    feed_dict ={X:test_data, keep_prob: 1.0 }
    generated_data, z_value = sess.run([X_out,z], feed_dict = feed_dict)
    sess.close()

    return generated_data,z_value

def plot_and_save(X_true,X_out,x,y,test_labels,n):
    if FLAGS.n_z ==2:
        ######################################################################
        def plot_manifold_canvas(n):
            x = np.linspace(-2, 2, n)
            y = np.linspace(-2, 2, n)
            canvas = np.empty((n * 28, n * 28))

            sess = tf.Session()
            saver = tf.train.Saver()
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, "parameter" + str(FLAGS.num_epoch - 1) + ".ckpt")

            for i, yi in enumerate(x):
                for j, xi in enumerate(y):
                    z = tf.constant([[xi, yi]] * FLAGS.batch_size, tf.float32)
                    output = VAE_model.Bernoulli_decoder(z, FLAGS.n_hidden_decoder, 28 * 28, 1.0)
                    temp = sess.run(output)
                    canvas[(n - i - 1) * 28: (n - i) * 28, j * 28: (j + 1) * 28] = temp[0].reshape(28, 28)
            plt.figure(figsize=(8, 8))
            plt.imshow(canvas, cmap="gray")
            plt.savefig("picture3")
            sess.close()
            ######################################################################
        plot.plot_images(X_true,X_out)
        plot.plot_2d_scatter(x,y,test_labels)
        plot_manifold_canvas(n)
    else:
        plot.plot_images(X_true,X_out)

if __name__=='__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    train()
    generated_data, z_value = test()
    plot_and_save(test_data, generated_data, z_value[:,0], z_value[:,1], test_labels, 20)
