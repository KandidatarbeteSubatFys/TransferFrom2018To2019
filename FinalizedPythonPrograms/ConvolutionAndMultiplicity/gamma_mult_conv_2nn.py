import tensorflow as tf
import time as t
import numpy as np
import sys
from scipy.sparse import find
#import matplotlib.pyplot as plt


# This function randomly selects some rows from the matrices batch_x and batch_y and returns the rows as two matrices.
# The number of rows is determined by the batch_size variable.
def gen_sub_set(batch_size, batch_x, batch_y):
    if not len(batch_x) == len(batch_y):
        raise ValueError('Lists most be of same length /Pontus')
    index_list = np.random.randint(0, len(batch_x), size=batch_size)
    return [batch_x[index] for index in index_list], [batch_y[index] for index in index_list]

def randomize_content(*args):
    # Randomize the internal order of two lists but still keep the relation between the events in the different files.
    tmp=len(args[0])
    for i in range(1,len(args)):
        if len(args[i])!= tmp:
            raise TypeError('Batches most be of same size')

    index_list = np.arange(0,len(args[0]))
    np.random.shuffle(index_list)
    return tuple(args[i][index_list] for i in range(len(args)))


def conv2d_first(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,7,1], padding='VALID')

def conv2d_rest(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,7,1], padding='VALID')




def get_y_for_specified_layers_and_nodes(x, number_of_hidden_layers, number_of_nodes_per_hidden_layer,number_particles, nr_filters, keep_rate):
    if number_of_hidden_layers==0:
        return -1
    weights={}
    biases={}
    weights["W" + str(1)] = tf.Variable(tf.truncated_normal([162 * nr_filters, number_of_nodes_per_hidden_layer], stddev=0.1), dtype=tf.float32) # 14 is the number of second pool crystals
    biases["b" + str(1)] = tf.Variable(tf.ones([number_of_nodes_per_hidden_layer]), dtype=tf.float32)
    for i in range(1,number_of_hidden_layers):
        weights["W"+str(i+1)]=tf.Variable(tf.truncated_normal([number_of_nodes_per_hidden_layer, number_of_nodes_per_hidden_layer], stddev=0.1), dtype=tf.float32)
        biases["b"+str(i+1)]=tf.Variable(tf.ones([number_of_nodes_per_hidden_layer]), dtype=tf.float32)
    weights["W" + str(number_of_hidden_layers+1)] = tf.Variable(tf.truncated_normal([number_of_nodes_per_hidden_layer, number_particles + 1], stddev=0.1),dtype=tf.float32)
    biases["b" + str(number_of_hidden_layers+1)] = tf.Variable(tf.ones([number_particles + 1]), dtype=tf.float32)
    y = x
    for i in range(number_of_hidden_layers):
        y=tf.nn.dropout(tf.nn.relu(tf.matmul(y, weights["W"+str(i+1)]) + biases["b"+str(i+1)]), keep_prob=keep_rate)
    y=tf.matmul(y,weights["W"+str(number_of_hidden_layers+1)]) + biases["b"+str(number_of_hidden_layers+1)]
    return y


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  #initial = tf.ones(shape=shape) #for testing
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  #initial = tf.ones(shape=shape) #for testing
  return tf.Variable(initial)


def max_accuracy(eval_accuracy_list):
    len_interval=50000
    interval_list=int(len_interval/100) # writes to list every 100 training iterations
    flipped_acc_list=np.flip(eval_accuracy_list, 0)
    number_of_items=len(flipped_acc_list)
    range_for_loop=int(number_of_items/interval_list)
    out=0
    for i in range(range_for_loop):
        acc_interval=np.average(flipped_acc_list[i*interval_list:(i+1)*interval_list])
        if acc_interval>out:
            out=acc_interval
    return out



def main(data_npz, number_particles, number_of_hidden_layers, number_of_nodes_per_hidden_layer, nr_filters_first_conv, keep_rate_input, comp):
    print('Reading convlolution and pooling matrices')
    conv_matrices = np.load('conv3_matrices.npz')
    first_conv_mat = conv_matrices['first_conv_mat']

    sparse_info_first_conv_mat = find(first_conv_mat)
    indices_sparse_transponate_first_conv_mat = []
    values_sparse_first_conv_mat = []
    dense_shape_sparse_first_conv_mat = [len(first_conv_mat[0]), len(first_conv_mat)]
    for i in range(len(sparse_info_first_conv_mat[0])):
        indices_sparse_transponate_first_conv_mat.append(
            [sparse_info_first_conv_mat[1][i], sparse_info_first_conv_mat[0][i]])
        values_sparse_first_conv_mat.append(1.0)
    sparse_transponate_first_conv_mat = tf.cast(
        tf.SparseTensor(indices_sparse_transponate_first_conv_mat,
                        values_sparse_first_conv_mat,
                        dense_shape_sparse_first_conv_mat), dtype=tf.float32)



    indicies_for_crystals_with_5_neighbours = [8, 14, 41, 49, 74, 78, 83, 87, 112, 120, 147, 153]
    zeros_to_ones_vector_first = np.zeros((1,162*7), dtype=np.float32)
    for i in range(len(indicies_for_crystals_with_5_neighbours)):
        zeros_to_ones_vector_first[0,7*indicies_for_crystals_with_5_neighbours[i] + 5] = 1

    zeros_to_ones_vector_rest = np.zeros((1, nr_filters_first_conv*162*7), dtype=np.float32)
    for i in range(len(zeros_to_ones_vector_rest)):
        zeros_to_ones_vector_rest[0, nr_filters_first_conv*7*indicies_for_crystals_with_5_neighbours[i] + 32*5 : nr_filters_first_conv*7*indicies_for_crystals_with_5_neighbours[i] + 32*6] = np.ones(32, dtype=np.float32)

    first_conv_mat_after_first_conv = np.zeros([162 * nr_filters_first_conv, nr_filters_first_conv * 7*162], dtype=np.float32)
    for i in range(len(first_conv_mat[0])):
        index_ordinary = first_conv_mat[:, i].argmax()
        for j in range(nr_filters_first_conv):
            if first_conv_mat[:, i].max() == 0:
                break
            first_conv_mat_after_first_conv[nr_filters_first_conv * index_ordinary + j, i * nr_filters_first_conv + j] = 1

    sparse_info_first_conv_mat_after_first_conv = find(first_conv_mat_after_first_conv)
    indices_sparse_transponate_first_conv_mat_after_first_conv = []
    values_sparse_first_conv_mat_after_first_conv = []
    dense_shape_sparse_first_conv_mat_after_first_conv = [len(first_conv_mat_after_first_conv[0]), len(first_conv_mat_after_first_conv)]
    for i in range(len(sparse_info_first_conv_mat_after_first_conv[0])):
        indices_sparse_transponate_first_conv_mat_after_first_conv.append([sparse_info_first_conv_mat_after_first_conv[1][i], sparse_info_first_conv_mat_after_first_conv[0][i]])
        values_sparse_first_conv_mat_after_first_conv.append(1.0)
    sparse_transponate_first_conv_mat_after_first_conv = tf.cast(tf.SparseTensor(indices_sparse_transponate_first_conv_mat_after_first_conv, values_sparse_first_conv_mat_after_first_conv, dense_shape_sparse_first_conv_mat_after_first_conv), dtype=tf.float32)

    # b = np.array([[0,0,1],[1,0,0],[1,0,0]], dtype=np.float32)
    # a = find(b)
    # print(a)
    # indices_sparse_transponate = []
    # values_sparse = []
    # dense_shape_sparse = [len(b), len(b[0])]
    # for i in range(len(a[0])):
    #     indices_sparse_transponate.append([a[1][i], a[0][i]])
    #     values_sparse.append(1.0)
    #
    # sparse_transponate = tf.cast(tf.SparseTensor(indices_sparse_transponate, values_sparse, dense_shape_sparse), dtype=tf.float32)
    # #sparse = tf.cast(tf.SparseTensor([[0,2],[1,0],[2,0]], [1.0, 1.0, 1.0], [3,3]), dtype=tf.float32)
    # input = np.array([[1,2,3],[4,5,6]], dtype=np.float32)
    # input_tensor = tf.convert_to_tensor(input, dtype=tf.float32)
    # print(sparse_transponate)
    # print(input_tensor)
    # sess_t = tf.Session()
    # print('sparse',sess_t.run(tf.transpose(tf.sparse_tensor_dense_matmul(sparse_transponate, tf.transpose(input_tensor)))))
    # print(np.matmul(input, b))
    # sess_t.close()
    # return

    #first_conv_mat = tf.convert_to_tensor(first_conv_mat, dtype=tf.float32)
    #first_conv_mat_after_first_conv = tf.convert_to_tensor(first_conv_mat_after_first_conv, dtype=tf.float32)
    zeros_to_ones_vector_first = tf.convert_to_tensor(zeros_to_ones_vector_first, dtype=tf.float32)
    zeros_to_ones_vector_rest = tf.convert_to_tensor(zeros_to_ones_vector_rest, dtype=tf.float32)

    print('Initializing variables')

    #Making placeholders for the inputdata (x) and the correct output data (y_)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 162]) #162=number of crystals in crystal ball detector
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, number_particles+1])
    keep_rate = tf.placeholder(dtype=tf.float32, shape=())

    W_conv1 = weight_variable([1, 7, 1, nr_filters_first_conv])
    b_conv1 = bias_variable([nr_filters_first_conv])
    W_conv2=weight_variable([1, 7, nr_filters_first_conv, nr_filters_first_conv])
    b_conv2=bias_variable([nr_filters_first_conv])


    conv_image_1 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_transponate_first_conv_mat ,tf.transpose(x))) #- zeros_to_ones_vector_first
    conv_image_1 = tf.reshape(conv_image_1, [-1, 1, 7*162, 1])
    h_conv1 = tf.nn.relu(conv2d_first(conv_image_1, W_conv1) + b_conv1)
    h_conv1_flat = tf.reshape(h_conv1,[-1,nr_filters_first_conv * 162])
    conv_image_2_flat = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_transponate_first_conv_mat_after_first_conv, tf.transpose(h_conv1_flat))) #- zeros_to_ones_vector_rest
    #conv_image_2_flat = tf.matmul(h_conv1_flat, first_conv_mat_after_first_conv) - zeros_to_ones_vector_rest
    conv_image_2 = tf.reshape(conv_image_2_flat, [-1, 1, 162*7, nr_filters_first_conv])
    h_conv2 = tf.nn.relu(conv2d_rest(conv_image_2, W_conv2) + b_conv2)

    from_conv = tf.reshape(h_conv2, [-1, 162*nr_filters_first_conv])


    y = get_y_for_specified_layers_and_nodes(from_conv, number_of_hidden_layers, number_of_nodes_per_hidden_layer,number_particles, nr_filters_first_conv, keep_rate)
    if isinstance(y, int):
        print("Error: number of hidden layers need to be at least one")
        return

    # As the loss funtion the softmax-crossentropy is used since it's common for classification problems.
    # To optimize the variables, Adam Optimizer is used since it fairs well in comparisons and is easy to use.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # To check the accuracy, the highest argument of the outputlayer and the one-hot-vector (one-hot is just a way to
    # represent the correct number of guns) is compared

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # converts boelean to ones and zeros and takes the mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    print('Reading data')
    data_set=np.load(data_npz)
    x_batch_train=data_set['x_batch_train']
    y_batch_train = data_set['y_batch_train']
    x_batch_eval = data_set['x_batch_eval']
    y_batch_eval = data_set['y_batch_eval']


    # Now the trainging begins. To get more information regarding the training part, and the whole program, see
    # "Deep learing for experts" on tensorflows webpage.
    print('Start training')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start = t.time()

    loss_list_train = []
    accuracy_list_eval = []
    accuracy_list_train = []
    iterations = []

    batch_size = 100
    # Number in "range"=number of training iterations
    for i in range(int(2e6)):
        # here 100 reandomly selected rows from the training set are extracted
        x_batch_sub, y_batch_sub = gen_sub_set(100, x_batch_train, y_batch_train)
        if i % 100 == 0:
            iterations.append(i)
            # from the 100 rows from the training set, the loss function is calculated
            loss_list_train.append(sess.run(loss, feed_dict={x: x_batch_sub, y_: y_batch_sub, keep_rate: 1}))
            # To calculate the accuracy, a bigger set of 300 rows are selected
            x_batch_eval_sub, y_batch_eval_sub = gen_sub_set(300, x_batch_eval, y_batch_eval)
            accuracy_value=sess.run(accuracy, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub, keep_rate: 1})
            accuracy_value_train=sess.run(accuracy, feed_dict={x: x_batch_sub, y_: y_batch_sub, keep_rate: 1})
            if i % 1000 == 0:
                print('Iteration nr. ', i, 'Acc: ', accuracy_value)
            accuracy_list_eval.append(accuracy_value)
            accuracy_list_train.append(accuracy_value_train)
        sess.run(train_step, feed_dict={x: x_batch_sub, y_: y_batch_sub, keep_rate: keep_rate_input})

    end=t.time()

    trainingtime = end - start

    # have to make sure that networks stops on representable position
    print('Stopping on representable loss')
    end_acc_list = []
    for _ in range(100):
        x_batch_sub, y_batch_sub = gen_sub_set(100, x_batch_train, y_batch_train)
        sess.run(train_step, feed_dict={x: x_batch_sub, y_: y_batch_sub, keep_rate: keep_rate_input})
        x_batch_eval_sub, y_batch_eval_sub = gen_sub_set(5000, x_batch_eval, y_batch_eval)
        end_acc_list.append(sess.run(accuracy, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub, keep_rate: 1}))
    end_acc_average = np.average(end_acc_list)

    found_average_acc = False
    for _ in range(500):
        x_batch_eval_sub, y_batch_eval_sub = gen_sub_set(5000, x_batch_eval, y_batch_eval)
        x_batch_sub, y_batch_sub = gen_sub_set(100, x_batch_train, y_batch_train)
        sess.run(train_step, feed_dict={x: x_batch_sub, y_: y_batch_sub, keep_rate: keep_rate_input})
        if abs(sess.run(accuracy, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub,
                                             keep_rate: 1}) - end_acc_average) < 0.005 * end_acc_average:
            found_average_acc = True
            break
    if not found_average_acc:
        print("Couldn't find average end acc")
    x_batch_eval_final, y_batch_eval_final = gen_sub_set(5000, x_batch_eval, y_batch_eval)
    final_acc = sess.run(accuracy, feed_dict={x: x_batch_eval_final, y_: y_batch_eval_final, keep_rate: 1})

    final_acc_avg_list = []
    for i in range(1000):
        x_batch_eval_sub, y_batch_eval_sub = gen_sub_set(100, x_batch_eval, y_batch_eval)
        final_acc_avg_list.append(
            sess.run(accuracy, feed_dict={x: x_batch_eval_sub, y_: y_batch_eval_sub, keep_rate: 1}))
    final_acc_avg = np.average(final_acc_avg_list)

    max_acc = max_accuracy(accuracy_list_eval)
    print('final acc: ', final_acc)
    print('final acc avg: ', final_acc_avg)
    print('max acc: ', max_acc)
    print('trainning time: ', trainingtime)

    np.savez(
        './report_mult_1_to_7_7e6input/2_nearest_neighbour_conv_layer/' + comp + 'mult_conv_2nn_' + str(number_particles) +'nr_particles_' + str(number_of_hidden_layers) + 'layers_' + str(number_of_nodes_per_hidden_layer) + 'nodes_' + str(
            nr_filters_first_conv) + 'filters_' + str(
            iterations[-1]) + 'its_' + str(keep_rate_input) + 'keep_rate_iterations_.npz', loss_list_train=np.array(loss_list_train),
        accuracy_list_train=np.array(accuracy_list_train), accuracy_list_eval=np.array(accuracy_list_eval),
        iterations=np.array(iterations), trainingtime=trainingtime, number_of_hidden_layers=number_of_hidden_layers,
        number_of_nodes_per_hidden_layer=number_of_nodes_per_hidden_layer, nr_filters_first_conv=nr_filters_first_conv, final_acc = final_acc, final_acc_avg = final_acc_avg,
        max_acc = max_acc)

    # print("Trainingtime: " + str(int(trainingtime))+" seconds")
    #
    # # Basic plotting of accuracy and training loss function using matplotlib.pyplot. Havn't figured out how to change the fontsize though.
    # fig, ax = plt.subplots(2, figsize=(20, 10)) #fig=entire figure, ax=subplots
    # ax[0].plot(iterations[0:-1], loss_list_train[0:-1])
    # ax[0].set(ylabel='Loss function', xlabel='Iteration')
    # ax[1].plot(iterations[0:-1], accuracy_list_train[0:-1])
    # ax[1].plot(iterations[0:-1], accuracy_list_eval[0:-1])
    # ax[1].set(ylabel='Accuracy', xlabel='Iterations')
    #
    # plt.show(fig)

if __name__ == '__main__':

    if len(sys.argv) == 5:
        layers = int(sys.argv[1])
        nodes = int(sys.argv[2])
        keep_rate = float(sys.argv[3])
        comp = sys.argv[4]
    elif len(sys.argv) == 1:
        layers = 2
        nodes = 2048
        keep_rate = 0.9
        comp = 'layers_nodes_comp/'
    else:
        raise ValueError('Number of arguments need to be 0 or 4')

    main('mult_data_set_buffert__og_up_to_6nr_of_partciles_in_eval_data_0.1_to_10MeV_7426789events_no_boost.npz', 7, layers, nodes, 32, keep_rate, comp)








