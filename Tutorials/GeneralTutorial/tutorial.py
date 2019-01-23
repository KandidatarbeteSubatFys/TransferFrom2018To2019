import numpy as np # numpy is efficient when dealing with fixed matricies
import tensorflow as tf # the tensorflow module is whats used for all neural network related things
import matplotlib.pyplot as plt # used for plotting
import sys # used for taking arguments when running script
from matplotlib.colors import LogNorm # useful norm for density plots


# Returns a randomly selected subset of batch_x and batch_y with length batch_size.
def gen_sub_set(batch_size, batch_x, batch_y):
    if not len(batch_x) == len(batch_y):
        raise ValueError('Lists most be of same length')
    indices = np.random.randint(0, len(batch_x), size=batch_size)
    return batch_x[indices], batch_y[indices]


# Returns the data in the textfile file, where you have to specify the number of data values per row
def read_data(file, columns):
    rows = file_len(file)
    out = np.zeros((rows, columns), dtype=np.float32)
    with open(file) as f:
        for i, line in enumerate(f):
            out[i] = np.fromstring(line.rstrip(), dtype=np.float32, sep=' ')
    return out


# Fast way to get number of rows in textfile
def file_len(file):
    with open(file) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# Could have implemented this in read_data with an if statement, but wanted to leave that pure for future use.
# This function is very limited and only works when you only allow events where three partclies are being fired.
# The function looks as each event at a time, and sorts each row (event) by reshaping the row into a 3x2 matrix.
# Then we sort based on the first column, and then reshape back to the 1x6 matrix.
def lowest_energy_first(gun_data):
    for i, row in enumerate(gun_data):
        reshaped_row = row.reshape((-1,2))
        reshaped_sorted_row = reshaped_row[reshaped_row[:, 0].argsort()]
        sorted_row = reshaped_sorted_row.flatten()
        gun_data[i] = sorted_row
    return gun_data.copy()



def main():


    print('Building network stucture')

    # x is a placeholder, which means that it is something we need to provide each training iteration. We will use
    # x as a placeholder for the input data, i.e. the deposited energy in the crystal. Since we're using the XB
    # detector, x gets 162 columns, one for each crystal. The None argument just denotes that our batch size, i.e.
    # the number of rows of the data we provide as the placeholder, can vary. As you will see, the matrix algebra that
    # will follow will not put any restrictions on the number of rows of x so we as mentioned we can use the None
    # argument to allow for diffirent batch sizes.
    x = tf.placeholder(dtype=tf.float32, shape=[None, 162])
    # y_ denotes the correct answer. Since we only allow events where 3 particles were fired in this tutorial,
    # y_ has 6 columns since each particle has an energy and a cos(theta). The batch size here can vary along with x
    # So we use None as the input for the number of rows.
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 6])

    # Now we're going to make our first hidden layer. Let's use 256 nodes per layer. First we need a weight matrix:
    W1 = tf.Variable(tf.truncated_normal([162, 256], stddev=0.1), dtype=tf.float32)
    # We want all of the elements in this matrix to be updated during the training, and to signal that to tensorflow
    # you make it a Variable. The tf.truncated_normal([162, 256], stddev=0.1) input is just the initial value,
    # and it's common to use normal distributed values. Now we multiply x with W1:
    y1_tmp1 = tf.matmul(x, W1)
    # If we have a batch size of e.g. 100, y1_tmp will get the size 100x256. So we note that if we for example
    # would have had 50 instead of 100 as the batch size, we would've gotten a size of 50x256, so that's the
    # reason why you can use None as the number of rows in the placeholers.
    # We will also need a bias vector, where all elements also should be trainable (common to set the initial
    # elemnts to 1).
    b1 = tf.Variable(tf.ones([256]), dtype=tf.float32)
    # Even though it isn't defined matemathically, we add b1 to y1_tmp:
    y1_tmp2 = y1_tmp1 + b1
    # The reason why we don't get an error is because of the fact that tensorflow allows broadcasting. b1 will
    # thus be added to every row of y1_tmp. To create the output of the first hidden layer, y1, we apply the
    # non-linear activation function relu:
    y1 = tf.nn.relu(y1_tmp2)

    # Let's have another layer with 128 nodes:
    W2 = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1), dtype=tf.float32)
    b2 = tf.Variable(tf.ones([128]), dtype=tf.float32)
    y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)

    # The last layer will be used to make sure that the size of output from the network (y) will have the same size
    # as the correct answer (y_), i.e. each event should produce 6 values:
    W3 = tf.Variable(tf.truncated_normal([128, 6], stddev=0.1), dtype=tf.float32)
    b3 = tf.Variable(tf.ones([6]), dtype=tf.float32)
    y = tf.nn.relu(tf.matmul(y2, W3) + b3)


    # Now we need to determine what we want the training process to minimize, i.e. the cost function.
    # We use cost/loss interchangeably.
    # To not complicate this tutorial, we use a rather simple approach. We will try to  minimize the mean square
    # error between the input and the output. However, to give the network a fair chance of reconstructing the particle
    # properties, we will provide the correct answer so that the info of the particle with the lowest energy always
    # is the first in the list, and then the rest follow in increasing energy order. So the first two element in each row
    # in the gun-dataset will be the energy and cos(theta) of the particle with the lowest energy, the 3rd and the 4th output elements
    # to energy and cos(theta) of the particle with the second lowest energy etc. To implement this, we simply
    # have to make sure that we change the gun data after we import it, so the the particle energy is in increasing order.
    # Just to be clear, WE NEVER USED THIS APPROACH FOR THE PLOTS IN THE REPORT, just wanted to keep things simple
    # here.
    loss = tf.reduce_mean(tf.square(tf.subtract(y_, y)))

    # Now we have to decide how we want to do the optimization. One common method is the optimizer Adam with step
    # length 1e-4:
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

    print('Reading and preparing data')

    # Before we actually train our variables W and b to produce a suitable network, we need to import some data to
    # train on. Remember that we need to change the order in the gun data so that the particles with the lowest
    # energies are first in order.
    det_data = read_data(sys.argv[1],162) # the number of crystals on crystal ball is 162
    gun_data = read_data(sys.argv[2],6) # only have 3 guns. Since we use data for both energy and angle, we get 6 columns

    # Here we fix the order
    gun_data_ordered= lowest_energy_first(gun_data)

    # We will want to use the majority of the data to train on, and then the remainder to evaluate on. Since
    # we won't change the network hyperparameters (e.g. nodes and layers) we don't need a third data set to
    # evaluate on, often called the test set.
    det_data_train = det_data[0:int(0.8*len(det_data))]
    det_data_eval = det_data[int(0.8 * len(det_data)):]
    gun_data_train = gun_data_ordered[0:int(0.8 * len(gun_data_ordered))]
    gun_data_eval = gun_data_ordered[int(0.8 * len(gun_data_ordered)):]



    print('Start training')
    # With the data ready, it is time to train. We need to define a session, otherwise we're unable to
    # evaluate e.g. variables. We also need to initialize the variables.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # will store values of the cost function as the training progresses
    loss_list_eval = []
    loss_list_train = []

    # We use 5e4 training iterations here, but we sometimes went with over 1e6 in some cases in the project.
    iterations = int(5e4)
    for i in range(iterations):
        # here 100 randomly selected rows from the training set are extracted.
        # It is usually advisable to keep the training batch small
        x_batch, y_batch = gen_sub_set(100, det_data_train, gun_data_train)
        if i % 100 == 0: # don't want to save values at every training step
            # Usually one uses a bigger batch size when evaluating, since we don't train on the evaluation
            # batch. So we extract 300 random events to evaluate on.
            x_batch_eval, y_batch_eval = gen_sub_set(300, det_data_eval, gun_data_eval)

            # Here we evaluate the cost function value using the evaluation data. To evaluate something that depends on
            # a placeholder, one must use the session and provide the data using feed_dict
            loss_value = sess.run(loss, feed_dict={x: x_batch_eval, y_: y_batch_eval})

            # Here we evaluate the cost function on the training data.
            loss_value_train = sess.run(loss, feed_dict={x: x_batch, y_: y_batch})

            if i % 1000 == 0:
                print('Iteration nr. ', i, 'Loss: ', loss_value)

            loss_list_eval.append(loss_value)
            loss_list_train.append(loss_value_train)

        # The actual training step is done here
        sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})

    # Now we have finished training the network. Since the training process is stochastic, the last step
    # could have resulted in a network that is worse than most of the networks from the previous 100 steps
    # for example. So if you're picky it's probably unwise to just select the last network, but we won't bother with
    # that in this tutorial.

    # Let's do some plotting over how the loss function changes over the training steps, and also over how well the
    # final network performs compare to the correct answer.
    print('Ploting')
    # Makes two subfigures
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # plot of how the cost function changes when evaluating the training data as well as on the eval data
    iter_list = [i for i in range(0, iterations, 100)]
    ax[0].plot(iter_list, loss_list_train)
    ax[0].plot(iter_list, loss_list_eval)
    ax[0].set(ylabel='Loss function (MeV)^2', xlabel='Iteration')
    ax[0].set_title('Evaluation cost function (orange) vs training cost function (blue)')

    # Reconstruction the evaluation data set by using the network:
    gun_data_from_network = sess.run(y, feed_dict={x: det_data_eval})
    # Here the particle energies are extracted from the data recontructed by the network, sice only plot the energies
    # in this tutorial
    reconstructed_energy_data_from_network = []
    for i in range(len(gun_data_from_network)):
        for j in range(int(len(gun_data_from_network[0])/2)):
            reconstructed_energy_data_from_network.append(gun_data_from_network[i][2*j])
    energy_data_correct = gun_data_eval[:, [0, 2, 4]].flatten().tolist()

    # 2d histogram, density plot, reconstructed vs correct energies
    h = ax[1].hist2d(energy_data_correct, reconstructed_energy_data_from_network , bins=int(np.sqrt(len(reconstructed_energy_data_from_network) / 2)), norm=LogNorm())
    ax[1].plot([0, 10], [0, 10], 'r')
    ax[1].set(xlabel='Gun energy (MeV)', ylabel='Predicted energy (MeV)')
    ax[1].set_title('Density plot of reconstructed vs correct energies')
    plt.colorbar(h[3], ax=ax[1])

    plt.show()

if __name__ == '__main__':
    main()
