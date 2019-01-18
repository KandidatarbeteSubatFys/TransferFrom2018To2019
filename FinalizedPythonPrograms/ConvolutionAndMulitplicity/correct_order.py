import numpy as np

# This function is used to get the neibours of each crystal, but where the first neighbour in each neighbour row is the
# one that is the closest to the beam-out crystal (crystal 81). The following neighbours come in an anticlockwise direction.
def correct_orientation_neighbours():
    geom_data=read_geom_txt()
    out=np.zeros((162,6),dtype=np.int)
    index = 0
    for i in geom_data:

        theta=np.zeros(6,dtype=np.float32)
        phi = np.zeros(6, dtype=np.float32)

        for j in range(6):
            row=int(i[5+j])
            if row != -1:
                theta[j]=np.array(geom_data[row-1][2]).astype(np.float32)
                phi[j]=np.array(geom_data[row-1][3]).astype(np.float32)
        theta_max_index=theta.argmax()

        max_theta=theta[theta_max_index]
        theta[theta_max_index]=0
        theta_second_largest_index=theta.argmax()
        theta_second_largest=theta[theta_second_largest_index]


        if max_theta-theta_second_largest<5.5:
            if phi[theta_max_index]>phi[theta_second_largest_index] and abs(phi[theta_second_largest_index]-phi[theta_max_index])<180:
                start_crystal=theta_max_index
            else:
                start_crystal=theta_second_largest_index
        else:
            start_crystal = theta_max_index
        if start_crystal==0:
            neighbour_row=i[5:]
        else:
            neighbour_row=i[5+start_crystal:]+i[5:5+start_crystal]
        out[index]=np.array(neighbour_row).astype(np.int)-1
        index+=1
    return out


# This function is used to turn the geometry file for XB into a matrix
def read_geom_txt():
    """"Reads the geometry file and removes white space"""
    out=[]
    with open('geom_xb.txt', 'r') as f:         # Open file
        for lines in f:
            tmp=lines.split('(')[1]
            tmp=tmp.split(')')[0]
            tmp=tmp.replace(' ','')
            tmp=tmp.split(',')
            out.append(tmp)
    return out


# This function gives the distance between two crystals (note: inputs are the indecies of the crystals, not the crystal number. So crystal number one has index 0 for example)
def distance_between_two_crystals(index1,index2):
    geom_data=read_geom_txt()
    theta_1=float(geom_data[index1][2])*np.pi/180
    phi_1=float(geom_data[index1][3])*np.pi/180
    theta_2 = float(geom_data[index2][2])*np.pi/180
    phi_2 = float(geom_data[index2][3])*np.pi/180

    r1=np.array([np.sin(theta_1)*np.cos(phi_1), np.sin(theta_1)*np.sin(phi_1), np.cos(theta_1)],dtype=np.float32)
    r2 = np.array([np.sin(theta_2) * np.cos(phi_2), np.sin(theta_2) * np.sin(phi_2), np.cos(theta_2)],dtype=np.float32)
    return np.linalg.norm(r1-r2)


# This index finds which of the crystals that is closest to the beam-out crystal (crystal 81, index 80)
def find_index_shortest_distance_to_crystal_81(min_dict):
    crystal_index=80
    tmp=np.ones(len(min_dict),dtype=np.float32)
    min_indecies=[]
    for i in range(len(min_dict)):
        tmp[i]=distance_between_two_crystals(crystal_index,min_dict[str(i+1)])
        min_indecies.append(min_dict[str(i+1)])

    return min_indecies[tmp.argmin()]


# Gets the next first pool crystal to add. When I say "first pool crystals" I mean the crystals of which the max pooling filter should be placed over.
# Technically the maxpool filter is placed over the convolution from one crystal.
def first_pool_index_for_next_crystal(crystal_index,list_neighbours,list_nn):
    d=3*np.ones(162,dtype=np.float32)
    for i in range(162):
        if i not in list_neighbours:   #went faster than writing "and" one time, and didn't care to investigate what I might be doing wrong
            if i not in list_nn:
                d[i]=distance_between_two_crystals(crystal_index,i)
    if sum(d)==3*162:
        return -1
    d_min_index={}
    for i in range(5):
        tmp_min=d.min()
        if tmp_min==3:
            break
        d_min_index[str(i+1)]=d.argmin()
        d[d_min_index[str(i+1)]]=3

    index=find_index_shortest_distance_to_crystal_81(d_min_index)

    return index


# This function gets the first pool crystals, i.e. the crystals of which the max pooling filter should be placed over.
def get_first_pool_crystals():
    geom_data=read_geom_txt()
    neighbour_index=correct_orientation_neighbours()
    list_nn=[80]
    list_neighbours=[]
    for i in range(1000):
        list_neighbours=list_neighbours+neighbour_index[list_nn[-1]].tolist()
        index_next_crystal=first_pool_index_for_next_crystal(list_nn[-1],list_neighbours,list_nn)
        if index_next_crystal==-1:
            break
        list_nn.append(index_next_crystal)
    return list_nn

# Here the neighbours for each first pool crystal is put is correct orientation. The first neighbour for each crystal is the one closest to
# the beam-out crystal (crystal 81 with index 80) and the rest follow anticlockwise.
def make_correct_oriented_first_pool_neighbours(not_oriented_first_pool_neighbour_matrix,first_pool_crystals):
    geom_data=read_geom_txt()
    out=np.ones(shape=not_oriented_first_pool_neighbour_matrix.shape,dtype=np.float32)*-1
    for i in first_pool_crystals:
        d=3
        for j in not_oriented_first_pool_neighbour_matrix[i]:
            if j!=-1:
                if distance_between_two_crystals(80,j)<d:
                    index_first_neighbour=j
                    d=distance_between_two_crystals(80,j)

        distances_to_first_neighbour=np.ones(6)*3
        for j in range(len(not_oriented_first_pool_neighbour_matrix[i])):
            if not_oriented_first_pool_neighbour_matrix[i,j]!=-1:
                distances_to_first_neighbour[j]=distance_between_two_crystals(index_first_neighbour,not_oriented_first_pool_neighbour_matrix[i,j])

        index=np.array([k for k in range(len(distances_to_first_neighbour))],dtype=np.int)
        index=index[distances_to_first_neighbour.argsort()]

        index_close_neighbour_1=not_oriented_first_pool_neighbour_matrix[i,index[1]]
        index_close_neighbour_2 = not_oriented_first_pool_neighbour_matrix[i,index[2]]
        if abs(float(geom_data[index_close_neighbour_1][3])-float(geom_data[index_close_neighbour_2][3]))<180:
            if float(geom_data[index_close_neighbour_1][3])<float(geom_data[index_close_neighbour_2][3]):
                index_oriented_close_neighbour=index_close_neighbour_1
            else:
                index_oriented_close_neighbour=index_close_neighbour_2
        else:
            if float(geom_data[index_close_neighbour_1][3]):
                index_oriented_close_neighbour=index_close_neighbour_1
            else:
                index_oriented_close_neighbour=index_close_neighbour_2
        list_processed_neighbours=[index_first_neighbour,index_oriented_close_neighbour]
        for j in range(6):
            index=-1
            d=3
            for k in not_oriented_first_pool_neighbour_matrix[i]:
                if k!=-1:
                    if k not in list_processed_neighbours: #didn't work one time whith "and" when using "not in" so haven't bother trying with that since then.
                        if distance_between_two_crystals(list_processed_neighbours[-1],k)<d:
                            d=distance_between_two_crystals(list_processed_neighbours[-1],k)
                            index=k
            if index==-1:
                break
            list_processed_neighbours.append(index)
            if j==0:
                list_processed_neighbours=list_processed_neighbours[1:]
        if len(list_processed_neighbours)==4:
            list_processed_neighbours.append(-1)
            list_processed_neighbours.append(-1)
        if len(list_processed_neighbours)==5:
            list_processed_neighbours.append(-1)
        out[i]=np.array(list_processed_neighbours)
    return out


# Here the neihbours for each first pool crystal is retrieved. They are in correct orientation (see make_correct_oriented_first_pool_neighbours regarding the orientation).
def make_neighbour_matrix_for_first_pool_crystals():
    first_pool_crystals=get_first_pool_crystals()
    geom_data=read_geom_txt()
    ordinary_neighbour_matrix=correct_orientation_neighbours()
    first_pool_neighbour_matrix=np.ones((162,6),dtype=np.int)*(-1)
    for i in first_pool_crystals:
        treated_neighbours=[i]
        ordinary_neighbours=ordinary_neighbour_matrix[i]
        for j in ordinary_neighbours:
            if j !=-1:
                for k in range(len(ordinary_neighbour_matrix)):
                    if k in first_pool_crystals:    #went slow when I used "and" one time, and didn't bother finding out what I was doing wrong
                        if k not in treated_neighbours:
                            if j in ordinary_neighbour_matrix[k]:
                                treated_neighbours.append(k)
        if len(treated_neighbours)==6:
            treated_neighbours.append(-1)
        if len(treated_neighbours)==5:
            treated_neighbours.append(-1)
            treated_neighbours.append(-1)
        first_pool_neighbour_matrix[i]=np.array(treated_neighbours[1:],np.int)

    correct_oriented_first_pool_neighbours=make_correct_oriented_first_pool_neighbours(first_pool_neighbour_matrix,first_pool_crystals)
    return correct_oriented_first_pool_neighbours.astype(np.int)


# Here the next second pool crystal is retrieved
def second_pool_index_for_next_crystal(crystal_index,list_neighbours,list_nn,first_pool_crystals):
    d=3*np.ones(162,dtype=np.float32)
    for i in first_pool_crystals:
        if i not in list_neighbours:   #goes faster than writing and, and didn't care to investigate what I might be doing wrong
            if i not in list_nn:
                d[i]=distance_between_two_crystals(crystal_index,i)
    if sum(d)==3*162:
        return -1
    d_min_index={}
    for i in range(5):
        tmp_min=d.min()
        if tmp_min==3:
            break
        d_min_index[str(i+1)]=d.argmin()
        d[d_min_index[str(i+1)]]=3 #just something bigger than 2

    index=find_index_shortest_distance_to_crystal_81(d_min_index)

    return index


# This functino gets the second pool crystals
def get_second_pool_crystals():
    first_pool_crystals=get_first_pool_crystals()
    neighbour_matrix_for_pool_crystals=make_neighbour_matrix_for_first_pool_crystals()

    list_processed_crystals = [80]
    list_neighbours = []
    for i in range(1000):
        list_neighbours = list_neighbours + neighbour_matrix_for_pool_crystals[list_processed_crystals[-1]].tolist()
        index_next_crystal = second_pool_index_for_next_crystal(list_processed_crystals[-1], list_neighbours, list_processed_crystals,first_pool_crystals)
        if index_next_crystal == -1:
            break
        list_processed_crystals.append(index_next_crystal)
    return list_processed_crystals


# Here you get the input crystals in the correct order, i.e. the order in which the second pool crystals
# comes first, the first pool crystals after that and then the rest. This way it becomes easier to to the first and second pooling,
# you just removes all inputs after a certain index.
def get_correct_order_input():
    second_pool_crystals=get_second_pool_crystals()
    first_pool_crystals=get_first_pool_crystals()
    all_crystals=[i for i in range(162)]
    correct_ordered_crystals=second_pool_crystals
    for i in first_pool_crystals:
        if i not in second_pool_crystals:
            correct_ordered_crystals.append(i)
    for i in all_crystals:
        if i not in correct_ordered_crystals:
            correct_ordered_crystals.append(i)
    return correct_ordered_crystals


# Based on the above fuctions, here the convolution image for 162 with different energies are made.
def make_conv_image1_1_event_correct_order(input_data):
    matrix_of_nearest_neighbours = correct_orientation_neighbours()
    correct_order_indices=get_correct_order_input()

    tmp = np.array([[], [], []],dtype=np.int)
    for j in correct_order_indices:
        tmp2 = np.zeros(3 * 3, dtype=np.float32)
        index = 0
        for k in matrix_of_nearest_neighbours[j]:
            if k == -1:
                tmp2[index] = 0
            else:
                tmp2[index] = input_data[k]
            index = index + 1
        tmp2[index] = input_data[j]
        tmp2 = tmp2.reshape(3, 3)
        tmp = np.concatenate((tmp, tmp2), axis=1)
    return tmp.astype(np.int)


# Here is the second convolution image made.
def make_conv_image2_1_event_correct_order(input_data):
    matrix_of_nearest_neighbours=make_neighbour_matrix_for_first_pool_crystals()
    first_pool_crystals_in_correct_order=np.array(get_correct_order_input()[0:len(get_first_pool_crystals())],dtype=np.int)
    tmp = np.array([[], [], []], dtype=np.int)
    for j in first_pool_crystals_in_correct_order:
        tmp2 = np.zeros(3 * 3, dtype=np.float32)
        index = 0
        for k in matrix_of_nearest_neighbours[j]:
            if k == -1:
                tmp2[index] = 0
            else:
                tmp2[index] = input_data[np.where(k==first_pool_crystals_in_correct_order)[0][0]]
            index = index + 1
        tmp2[index] = input_data[np.where(j==first_pool_crystals_in_correct_order)[0][0]]
        tmp2 = tmp2.reshape(3, 3)
        tmp = np.concatenate((tmp, tmp2), axis=1)
    return tmp.astype(np.int)


def get_neighbours_and_neighbours_neighbours_conv3():
    neighbour_matrix_correct_order = correct_orientation_neighbours()
    out = (-1000)*np.ones((len(neighbour_matrix_correct_order),18), dtype=np.int)
    for i in range(len(neighbour_matrix_correct_order)):
        dealt_with_list = [i] + neighbour_matrix_correct_order[i].tolist()
        for j in range(len(neighbour_matrix_correct_order[i])):
            neighbours_per_neighbour = []
            if neighbour_matrix_correct_order[i,j] != -1:
                for k in range(len(neighbour_matrix_correct_order[neighbour_matrix_correct_order[i,j]])):
                    if neighbour_matrix_correct_order[neighbour_matrix_correct_order[i,j],k] != -1:
                        if not neighbour_matrix_correct_order[neighbour_matrix_correct_order[i,j],k] in dealt_with_list:
                            dealt_with_list.append(neighbour_matrix_correct_order[neighbour_matrix_correct_order[i,j],k])
                            neighbours_per_neighbour.append(neighbour_matrix_correct_order[neighbour_matrix_correct_order[i,j],k])
                distances_to_previous_neighbour = []
                for k in neighbours_per_neighbour:
                    if j != 0:
                        distances_to_previous_neighbour.append(distance_between_two_crystals(k,neighbour_matrix_correct_order[i,j-1]))
                    else:
                        distances_to_previous_neighbour.append(distance_between_two_crystals(k,neighbour_matrix_correct_order[i, -1]))
                if len(neighbours_per_neighbour) != 0:
                    neighbours_per_neighbour = np.array(neighbours_per_neighbour)
                    distances_to_previous_neighbour = np.array(distances_to_previous_neighbour)
                    dealt_with_list[len(dealt_with_list)-len(neighbours_per_neighbour):] =  neighbours_per_neighbour[distances_to_previous_neighbour.argsort()].tolist()
        dealt_with_list = dealt_with_list[1:]
        dealt_with_list = dealt_with_list +(18-len(dealt_with_list))*[-1]
        out[i] = np.array(dealt_with_list,dtype=np.int)
    return out


def get_second_conv_matrix_conv3():
    neighbours = get_neighbours_and_neighbours_neighbours_conv3()
    second_pool_crystals = get_second_pool_crystals()
    number_of_second_pool_crystals = len(second_pool_crystals)

    out = np.zeros((162,19*number_of_second_pool_crystals),dtype=np.int)
    for i in range(number_of_second_pool_crystals):
        neighbour_row = neighbours[second_pool_crystals[i]]

        for j in range(len(neighbour_row)):
            if neighbour_row[j] == -1:
                neighbour_row = np.delete(neighbour_row, j)
                neighbour_row = np.concatenate((neighbour_row,np.array([-1])))
        final_row = np.concatenate((neighbour_row, np.array([second_pool_crystals[i]])))

        for j in range(len(final_row)):
            if final_row[j] != -1:
                out[final_row[j], i*len(final_row) +j] = 1

    return out


def get_first_conv_matrix_conv3():
    neighbours = correct_orientation_neighbours()

    out = np.zeros((162, 162*7), dtype=np.int)
    for i in range(len(neighbours)):
        neighbour_row = neighbours[i]
        for j in range(len(neighbour_row)):
            if neighbour_row[j] == -1:
                neighbour_row = np.delete(neighbour_row, j)
                neighbour_row = np.concatenate((neighbour_row, np.array([-1])))
        final_row = np.concatenate((neighbour_row, np.array([i])))

        for j in range(len(final_row)):
            if final_row[j] != -1:
                out[final_row[j], i*len(final_row) +j] = 1
    return out

def get_indicies_for_5_neighbours_conv4():
    neighbours = correct_orientation_neighbours()
    indicies = []
    for i in range(len(neighbours)):
        if min(neighbours[i]) == -1:
            indicies.append(i)
    return indicies


# If you run your tensorflow program on a gpu, it's better to use matrix multiplication than try to implement functions
# like the above to make the convolution images, so here the matrix that gives the first covolution image i made.
def get_first_conv_matrix():
    tmp=np.array([i for i in range(1,163)],dtype=np.int)
    out=np.zeros((162,9*162),dtype=np.int)
    conv_row=make_conv_image1_1_event_correct_order(tmp).flatten()
    for i in range(len(out[0])):
        if conv_row[i] != 0:
            out[conv_row[i]-1,i]=1
    return out


# Didn't end up using this because the final matrix in tensorflow became too large (there you have to account for that you
# have more than one event and more than one filter), but it removes all but the first pool crystals.
def get_matrix_to_remove_all_but_first_pool():
    out=np.zeros((162*3,3*len(get_first_pool_crystals())),dtype=np.int)
    for i in range(len(out[0])):
        out[i,i]=1
    return out


# Here the matrix that gives the second covolution image i made.
def get_second_conv_matrix():
    first_pool_crystals=get_first_pool_crystals()
    tmp=np.array([i for i in range(1,len(first_pool_crystals)+1)])
    out=np.zeros((len(first_pool_crystals),9*len(first_pool_crystals)),dtype=np.int)
    conv_row=make_conv_image2_1_event_correct_order(tmp).flatten()
    for i in range(len(out[0])):
        if conv_row[i] !=0:
            out[conv_row[i]-1,i]=1
    return out


# Didn't end up using this  but it removes all but the second pool crystals.
def get_matrix_to_remove_all_but_second_pool():
    out=np.zeros((len(get_first_pool_crystals())*3,3*len(get_second_pool_crystals())),dtype=np.int)
    for i in range(len(out[0])):
        out[i,i]=1
    return out


# The first convolution matrix gives the output in the "correct order", i.e. the order in which the second pool crystals
# comes first, the first pool crystals after that and then the rest, but the input has to be in the normal order, i.e. 0,1,2,3,4,..,161
# so after the first convolution you have to use the matrix produced by this function to make the input into the normal order.
def get_matrix_to_make_normal_order_after_first_conv():
    correct_order=np.array(get_correct_order_input(),dtype=np.int)
    out=np.zeros((162,162),dtype=np.int)
    for i in range(162):
        index=correct_order.argmin()
        correct_order[index]=200 # just something bigger than 161
        out[index,i]=1
    return out

if __name__=='__main__':
    # Use the functions above to create the matrices you need and preferably save them in an npz file with np.savez


    # # Below is just to get something I used to check whether the process in the tensorflow program worked.
    # # What it does more specifically is to do convolution over each crystal and its neighbours with 1 filter, maxpooling over neighbours with overlap (more specifically each crystal's
    # # neighbours shouldn't overlap more than at one place with the next pooling crystal, so after the first maxpooling you've decreased from 162 to 47 points),
    # # convolution again like the first (1 filter) time but now over the neighbours of the first pool crystals, and then maxpooling like the first time.
    # # Both the convolution filters has only ones as elements and the bias variable is also one.
    #
    np.set_printoptions(threshold=np.nan)
    first_pool_crystals=get_first_pool_crystals()
    fpc_get_index=np.zeros(162,dtype=np.int)
    index=0
    input=[i for i in range(1,163)]
    A=get_first_conv_matrix()
    conv_image_1_flat=np.matmul(input,A)
    conv_image_1=np.reshape(conv_image_1_flat,[3,3*162])
    print(conv_image_1)
    # h_conv1=np.zeros(162)
    # for i in range(162):
    #     h_conv1[i]=conv_image_1[:,3*i:3*(i+1)].sum()+1
    # B=get_matrix_to_make_normal_order_after_first_conv()
    # h_conv1=np.matmul(h_conv1,B)
    # pool_image_1_long_flat=np.matmul(h_conv1,A)
    # pool_image_1_long=np.reshape(pool_image_1_long_flat,[3,3*162])
    # pool_image_1=pool_image_1_long[:,0:3*47]
    # h_pool1=np.zeros(47)
    # for i in range(47):
    #     h_pool1[i]=pool_image_1[:,3*i:3*(i+1)].max()
    # print(h_pool1)
    # C=get_second_conv_matrix()
    # conv_image_2_flat=np.matmul(h_pool1,C)
    # print(conv_image_2_flat)
    #
    #
    # conv_image_2=np.reshape(conv_image_2_flat,[3,3*47])
    # h_conv2=np.zeros(47)
    # for i in range(47):
    #     h_conv2[i]=conv_image_2[:,3*i:3*(i+1)].sum()+1
    # pool_image_2_long_flat=np.matmul(h_conv2,C)
    # pool_image_2_long=np.reshape(pool_image_2_long_flat,[3,3*47])
    # pool_image_2=pool_image_2_long[:,0:14*3]
    # h_pool2=np.zeros(14)
    # for i in range(14):
    #     h_pool2[i]=pool_image_2[:,3*i:3*(i+1)].max()
    # print(h_pool2)

    #print(correct_orientation_neighbours())
   #np.set_printoptions(threshold=np.nan)
    #print(get_first_pool_crystals())
    #test = np.arange(1,163,dtype=np.int)


    # second_conv_matrix_conv3 = get_second_conv_matrix_conv3()
    # first_conv_matrix_conv3 = get_first_conv_matrix_conv3()
    #
    # conv_image_1 = np.matmul(test,first_conv_matrix_conv3)
    # h_conv_1 = np.ones(162, dtype=np.int)*1000
    # for i in range(int(len(conv_image_1)/7)):
    #     tmp = np.sum(conv_image_1[7*i:7*(i+1)])
    #     h_conv_1[i] =tmp
    # h_conv_1 = h_conv_1 +1
    # conv_image_2 = np.matmul(h_conv_1, second_conv_matrix_conv3)
    # h_conv_2 = np.ones(14, dtype=np.int) * 1000
    # for i in range(int(len(conv_image_2) / 19)):
    #     tmp = np.sum(conv_image_2[19 * i:19 * (i + 1)])
    #     h_conv_2[i] = tmp
    # #h_conv_2 = h_conv_2 +1
    # print(2*h_conv_2+1)









