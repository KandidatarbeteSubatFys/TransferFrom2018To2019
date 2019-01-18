import numpy as np


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


# This function is used to get the neighbours of each crystal, but not in the same order as they are in the
# XB geometry file. Instead the first neighbour in each neighbour row is the one that is the closest to
# the beam-out crystal (crystal 81) and the following neighbours come in an anticlockwise direction.
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


# The convolution process we would like to implement is where each filter slides over every crystal and covers
# their neighbours. Having the 162 measured energies, we create a 162*7 long vector (each crystal can have up to
# 6 neighbours) where the first element is the energy deposited in crystal 1, and the 6 following values are the
# energies of the neighbouring crystals of crystal 1. The order of the 6 neighbours is set to be so that the
# first neighbour is the one closest to crystal 81, and the following come in anti-clockwise direction. The second
# group of 7 elements in the 162*7 vector contains the energy of crystal 2 and its neighbours and so on. So if we
# slide over this 162*7 long vector with a convolution filter of size 1x7 with stride 7 (i.e. no overlap), we  have
# implemented the convolution process we wanted: we slide each filter over each crystal with constant orientation.

# Since matrix multiplication is very efficient in tensorflow compared to other operations,
# we use matrix multiplication to go from the 162 vector to the 162*7 vector, and the matrix we multiply
# the 162 vector with is created in this function.
def get_nearest_neighbour_conv_matrix():
    neighbours = correct_orientation_neighbours()

    out = np.zeros((162, 162*7), dtype=np.int)
    for i in range(len(neighbours)):
        neighbour_row = neighbours[i]
        for j in range(len(neighbour_row)):
            if neighbour_row[j] == -1: # some crystals don't have 6 neighbours, is indicated eith -1, so fills out with zeros
                neighbour_row = np.delete(neighbour_row, j)
                neighbour_row = np.concatenate((neighbour_row, np.array([-1])))
        final_row = np.concatenate((neighbour_row, np.array([i])))

        for j in range(len(final_row)):
            if final_row[j] != -1:
                out[final_row[j], i*len(final_row) +j] = 1
    return out


if __name__=='__main__':
    np.save('tutorial_conv_mat', get_nearest_neighbour_conv_matrix()) # saving the convolution matrix







