import numpy as np # numpy is efficient when dealing with fixed matricies
import tensorflow as tf # the tensorflow module is whats used for all neural network related things
import matplotlib.pyplot as plt # used for plotting
import sys # used for taking arguments when running script
from matplotlib.colors import LogNorm # useful norm for density plots
from scipy.sparse import find # needed to get info on how to info of matrix so that we can make a sparse version of it


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


# Fast way to get number of rows in textfile (props to Google)
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
        reshaped_sorted_row = reshaped_row[reshaped_row[:, 1].argsort()]
        sorted_row = reshaped_sorted_row.reshape((1,-1))
        gun_data[i] = sorted_row
    return gun_data

# This function we use for the weighting when doing convolution. x is the input and W is the weight matrix.
# The W matrix will stride across x with step length 1 in all 4 directions except in the y direction where the stride is 7 
# (stride input is [stride_length_in_z, stride_length_in_x, stride_length_in_y, stride_length_in_channels], but was a while
# since I did this so double check meaning of dimension 1 and 4 (they will both be one here anyway so a mix up doesn't
# matter here)).
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,7,1], padding='VALID')


def main():
    print('Building network stucture')

    # x is a placeholder, which means that it is something we need to provide each training iteration. We will use
    # x as a placeholder for the input data, i.e. the deposited energy in the crystal. Since we're using the XB
    # detector, x gets 162 columns, one for each crystal. The None argument just denotes that our batch size, i.e.
    # the number of rows of the data we provide as the placeholder, can vary. As you will see, the matrix algebra that will follow will not
    # put any restrictions on the number of rows of x so we as mentioned we can use the None argument to allow for diffirent batch sizes.
    x = tf.placeholder(dtype=tf.float32, shape=[None, 162])
    # y_ denotes the correct answer. Since we only allow events where 3 particles were fired in this tutorial,
    # y_ has 6 columns since each particle has an energy and a cos(theta). The batch size here can vary along with x
    # So we use None as the input for the number of rows.
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 6])

    #### The difference between this script and the tutorial.py one, is just this first part of the main function
    # and the functions above that we utilize here. ####
    print('Reading convlolution matrices')

    conv_matrix= np.load('tutorial_conv_mat.npy')
    nr_filters_first_conv_layer = 8
    nr_filters_second_conv_layer = 16

    # The follwoing here is a messy, but it's not very important to know the details initially since it only is important
    # for the data formatting, and not for the conceptual understanding of the method. As we will see later,
    # to go from the [batch_size, 1, 162, 8] sized output from the first convolution layer to the size
    # [batch_size, 1, 162*7, 8] using the conv_matrix and not slowing down the training process can be tricky.
    # So what is done here is to flatten it out and then multiply it with big sparse matrix, and it's that sparse
    # matrix that is built here. Should have put this in a separate function, but not in the mood of fixing that
    # now;) Anyhow, what you want to implement is not hard to graps (see conv tutorial pdf)
    # so the technical details for the sparse matrices are not super important.

###### Messy and not important for general understanding, should've preferably been in a separate function #####
    # Sparse matrix for first conv layer. It's not that important to have this matrix as sparse, but
    # can be usefull if you want to have many filters.
    sparse_info_first_conv_mat = find(conv_matrix)
    indices_sparse_transponate_first_conv_mat = []
    values_sparse_first_conv_mat = []
    dense_shape_sparse_first_conv_mat = [len(conv_matrix[0]), len(conv_matrix)]
    for i in range(len(sparse_info_first_conv_mat[0])):
        indices_sparse_transponate_first_conv_mat.append(
            [sparse_info_first_conv_mat[1][i], sparse_info_first_conv_mat[0][i]])
        values_sparse_first_conv_mat.append(1.0)
    sparse_transponate_first_conv_mat = tf.cast(
        tf.SparseTensor(indices_sparse_transponate_first_conv_mat,
                        values_sparse_first_conv_mat,
                        dense_shape_sparse_first_conv_mat), dtype=tf.float32)

    # Sparse matrix for second conv layer. If I recall correctly it was essential that you made this sparse.
    first_conv_mat_after_first_conv = np.zeros([162 * nr_filters_first_conv_layer, nr_filters_first_conv_layer * 7 * 162],
                                               dtype=np.float32)
    for i in range(len(conv_matrix[0])):
        index_ordinary = conv_matrix[:, i].argmax()
        for j in range(nr_filters_first_conv_layer):
            if conv_matrix[:, i].max() == 0:
                break
            first_conv_mat_after_first_conv[
                nr_filters_first_conv_layer * index_ordinary + j, i * nr_filters_first_conv_layer + j] = 1

    sparse_info_first_conv_mat_after_first_conv = find(first_conv_mat_after_first_conv)
    indices_sparse_transponate_first_conv_mat_after_first_conv = []
    values_sparse_first_conv_mat_after_first_conv = []
    dense_shape_sparse_first_conv_mat_after_first_conv = [len(first_conv_mat_after_first_conv[0]),
                                                          len(first_conv_mat_after_first_conv)]
    for i in range(len(sparse_info_first_conv_mat_after_first_conv[0])):
        indices_sparse_transponate_first_conv_mat_after_first_conv.append(
            [sparse_info_first_conv_mat_after_first_conv[1][i], sparse_info_first_conv_mat_after_first_conv[0][i]])
        values_sparse_first_conv_mat_after_first_conv.append(1.0)
    sparse_transponate_first_conv_mat_after_first_conv = tf.cast(
        tf.SparseTensor(indices_sparse_transponate_first_conv_mat_after_first_conv,
                        values_sparse_first_conv_mat_after_first_conv,
                        dense_shape_sparse_first_conv_mat_after_first_conv), dtype=tf.float32)
########### End of messy and boring part ###########


    # To create the weight matrix for the first convolution layer we do the same thing as we do for fully-
    # connected layers, but the size now is 1x7x1x8. The last dimension weill determine the number of filters
    # and 1x7x1 will determine the size of each filter. Each filter has a bias, so the bias vector has shape 1x8.
    W_conv1_shape = [1, 7, 1, nr_filters_first_conv_layer]
    W_conv1 = tf.Variable(tf.truncated_normal(W_conv1_shape, stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[nr_filters_first_conv_layer]))

    # The weight matrix for the second filter is produced as the first layer, but note now that each filter also
    # has a length 8 in the 3rd dimension. See the pdf for an explanation for this.
    W_conv2_shape = [1, 7, nr_filters_first_conv_layer, nr_filters_second_conv_layer]
    W_conv2 = tf.Variable(tf.truncated_normal(W_conv2_shape, stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[nr_filters_second_conv_layer]))

    # Here it may appear a bit messy, but what we do here is to build the 162*7 long row-vector that we will do
    # the convolution on (see pdf) and we call this vector conv_image_1. The tf.sparse_tensor_dense_matmul(A,B
    # does A*B where A is sparse and B is dense. Since we want to do x*C where x is dense and C is
    # sparse, we do this with tf.sparse_tensor_dense_matmul(C^T,x^T)^T.
    conv_image_1 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_transponate_first_conv_mat, tf.transpose(x)))
    conv_image_1 = tf.reshape(conv_image_1, [-1, 1, 7 * 162, 1])
    # The output from conv layer one is h_conv1, which will have size: batchsize x 1 x 162 x 8, where 8 comes from
    # the number of filters used. For every one of the 8 filter outputs, we want to multiply it with the conv_matrix
    # The best solution for this I came up with was to flatten h_conv1, multiply it with the big sparse matrix I built
    # above, and then reshape the output to get the desired result.
    # It of course looks a bit messy, but it does the job.
    h_conv1 = tf.nn.relu(conv2d(conv_image_1, W_conv1) + b_conv1)
    h_conv1_flat = tf.reshape(h_conv1, [-1, nr_filters_first_conv_layer * 162])

    # Using the sparse matrix created for the second conv layer, we get the conv_image_2.
    conv_image_2_flat = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_transponate_first_conv_mat_after_first_conv,
                                                                   tf.transpose(h_conv1_flat)))
    conv_image_2 = tf.reshape(conv_image_2_flat, [-1, 1, 162 * 7, nr_filters_first_conv_layer])
    # The output from conv layer 2 is given by using W_conv2 and b_conv2
    h_conv2 = tf.nn.relu(conv2d(conv_image_2, W_conv2) + b_conv2)

    # The input to the ordinary fully-connected network will just be the output from conv layer 2, but flattened.
    from_conv = tf.reshape(h_conv2, [-1, 162 * nr_filters_second_conv_layer])

    #### The following is just same as the tutorial.py script, but with conv_out instead of x.
    # Except for that we after the trainng couldn't evaulate the the evaluation data set at once due to our matrix
    # multiplication method  causing memeory error.####

    # Now we're going to make our first hiddenfully connected hidden layer. Let's use 256 nodes per layer.
    # First need a weight matrix:
    W1 = tf.Variable(tf.truncated_normal([162*nr_filters_second_conv_layer, 256], stddev=0.1), dtype=tf.float32)
    # We want all of the elements in this matrix to be updated during the training, and to signal that to tensorflow
    # you make it a Variable. The tf.truncated_normal([162*16, 256], stddev=0.1) input is just the initial value,
    # and it's common to use normal distributed values will mean 0. Now we multiply x with W1:
    y1_tmp1 = tf.matmul(from_conv, W1)
    # If we have a batch size of e.g. 100, y1_tmp will get the size 256x100. So we note that if we for example
    # would have had 50 instead of 100 as the batch size, we would've gotten a size of 256x50, so that's the
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

    # Now we have to decide HOW we wan't to minimize. One common method is the optimizer Adam:
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

    print('Reading and preparing data')

    # Before we actually train our variables W and b to produce a suitable network, we need to import some data to
    # train on. Remember that we need to change the order in the gun data so that the particles with the lowest
    # energies are first in order.
    det_data = read_data(sys.argv[1],162) # number of crystals on crystal ball
    gun_data = read_data(sys.argv[2],6) # only have 3 guns. Since we use data for both energy and angle, we get 6 columns

    gun_data= lowest_energy_first(gun_data)

    # We will want to use the majority of the data to train on, and then the remainder to evaluate on. Since
    # we won't change the network hyperparameters (e.g. nodes and layers) we don't need a third data set to
    # evaluate on, ofter called the test set.
    det_data_train = det_data[0:int(0.8*len(det_data))]
    det_data_eval = det_data[int(0.8 * len(det_data)):]
    gun_data_train = gun_data[0:int(0.8 * len(gun_data))]
    gun_data_eval = gun_data[int(0.8 * len(gun_data)):]



    print('Start training')
    # With the data ready, it is time to train. We need to define a session, otherwise we're unable to
    # evaluate e.g. variables. We also need to initialize the variables.
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # will store values of the cost function as the training progresses
    loss_list_eval = []
    loss_list_train = []

    iterations = int(5e4)
    for i in range(iterations):
        # here 100 randomly selected rows from the training set are extracted.
        # It is usually advicable to keep the training batch small
        x_batch, y_batch = gen_sub_set(100, det_data_train, gun_data_train)
        if i % 100 == 0: # don't want to save values at every training step
            # Usually one uses a bigger batch size when evaluating, since we don't train on based on the evaluation
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
    # could have resulted in a network that is much worse than most of the networks from the previous 100 steps
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

    # Runs the network on some of the data set (all at once caused memory error
    gun_data_from_network = sess.run(y, feed_dict={x: det_data_eval[0:10000]})
    # Here the particle energies are extracted from the data recontructed by the network. We only plot the energies
    # in this tutorial
    energy_data_from_network = []
    for i in range(len(gun_data_from_network)):
        for j in range(int(len(gun_data_from_network[0])/2)):
            energy_data_from_network.append(gun_data_from_network[i][2*j])
    energy_data_correct = gun_data_eval[0:10000, [0, 2, 4]].flatten().tolist()

    # 2d histogram, density plot, reconstructed vs correct energies
    h = ax[1].hist2d(energy_data_correct, energy_data_from_network, bins=int(np.sqrt(len(energy_data_from_network) / 2)), norm=LogNorm())
    ax[1].plot([0, 10], [0, 10], 'r')
    ax[1].set(xlabel='Gun energy (MeV)', ylabel='Predicted energy (MeV)')
    ax[1].set_title('Density plot of reconstructed vs correct energies')
    plt.colorbar(h[3], ax=ax[1])

    plt.show()

if __name__ == '__main__':
    main()
