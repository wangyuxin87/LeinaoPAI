from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os


def main():

    mnistdata_dir = '/gdata/MNIST'
    checkpoint_dir = '/userhome/checkpoints'
    checkpoint_last = os.path.join(checkpoint_dir, "mnist_last")
    checkpoint_best = os.path.join(checkpoint_dir, "mnist_best")

    mnist = input_data.read_data_sets(mnistdata_dir, one_hot=True)

    batch_size = 64
    train_num = mnist.train._num_examples
    epochs = 10

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    best_acc = load_checkpoint(
        sess, checkpoint_last, saver, accuracy, cross_entropy, x, y_, keep_prob)

    for epoch in range(0, epochs):
        train_one_epoch(mnist, train_num, batch_size, accuracy,
                        cross_entropy, train_step, epoch, x, y_, keep_prob)

        acc, _ = val_one_epoch(
            mnist, accuracy, cross_entropy, x, y_, keep_prob, epoch)

        is_best = acc > best_acc
        if is_best:
            best_acc = acc

        save_checkpoint(sess, checkpoint_last, saver,
                        checkpoint_best, is_best=is_best)


def train_one_epoch(mnist, train_num, batch_size, accuracy, cross_entropy,
                    train_step, epoch, x, y_, keep_prob):
    total_accuracy = 0
    total_loss = 0
    for i in range(train_num // batch_size):
        batch = mnist.train.next_batch(batch_size)

        batch_accuracy = accuracy.eval(
            feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        total_accuracy += batch_accuracy
        train_loss = cross_entropy.eval(
            feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        total_loss += train_loss

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("Epoch {} - Training: Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(epoch, total_accuracy / (train_num // batch_size),
                  total_loss / train_num))


def val_one_epoch(mnist, accuracy, cross_entropy, x, y_, keep_prob, epoch=-1):
    val_accuracy = accuracy.eval(
        feed_dict={x: mnist.validation.images,
                   y_: mnist.validation.labels,
                   keep_prob: 1.0})
    val_loss = cross_entropy.eval(
        feed_dict={x: mnist.validation.images,
                   y_: mnist.validation.labels,
                   keep_prob: 1.0})
    val_num = mnist.validation._num_examples
    val_loss = val_loss / val_num

    if epoch != -1:
        print("Epoch {} - Validation: Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(epoch, val_accuracy, val_loss))
    return val_accuracy, val_loss


def save_checkpoint(sess, filename, saver, checkpoint_best, is_best=0):
    if not os.path.exists('/userhome/checkpoints'):
        os.makedirs('/userhome/checkpoints')

    saver.save(sess, filename)
    if is_best:
        saver.save(sess, checkpoint_best)


def load_checkpoint(sess, filename, saver, accuracy,
                    cross_entropy, x, y_, keep_prob):
    try:
        print("Loading checkpoint '{}'".format(filename))
        saver.restore(sess, filename)
        avg_accuracy, avg_loss = val_one_epoch(accuracy, cross_entropy,
                                               x, y_, keep_prob)

        print("Checkpoint validation Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(avg_accuracy, avg_loss))
        return avg_accuracy

    except:
        print("No specified checkpoint exists. Skipping...")
        print("**First time to train**")
        return 0


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    main()
