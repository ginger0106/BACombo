import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator


def parse_distribution_info(x_train,y_train,data_info_file,shuffle=True):

    file_lines = open(data_info_file,"r").readlines()
    num_label = len(y_train[0])
    dist_vectors = []

    node_partition_list = []
    for line in file_lines:
        temp_list = []
        line = line.rstrip()
        for item in line.split(" "):
            temp_list.append(float(item))
        assert len(temp_list) == num_label
        node_partition_list.append(temp_list)

    for label_partition_list in np.array(node_partition_list).transpose():
        assert np.sum(label_partition_list) <= 100.0

    num_nodes = len(node_partition_list)

    label_dict = {}
    for i in range(len(x_train)):
        if int(y_train[i].argmax()) in label_dict:
            # label_dict.\
            #
            #     has_key(int(y_train[i].argmax())):
            label_dict[int(y_train[i].argmax())].append((x_train[i],y_train[i]))
        else:
            label_dict[int(y_train[i].argmax())] = [(x_train[i],y_train[i])]


    labels = label_dict.keys()
    quantity_dict = {}
    current_index = [0] * num_label
    for l in labels:
        quantity_dict[l] = len(label_dict[l])
        if shuffle:
            random.shuffle(label_dict[l])
            label_dict[l] = label_dict[l]
    node_partition_result = []
    intervals = calculate_intervals(node_partition_list,quantity_dict)

    for node in range(num_nodes):
        node_dataset = []
        node_dist_vec = []
        for portion,l in zip(node_partition_list[node],labels):
            portion_length = int(len(label_dict[l]) * float(portion) / 100.0)
            node_dist_vec.append(portion_length)
            node_dataset.extend(label_dict[l][current_index[l]:current_index[l] + portion_length])
            current_index[l] = current_index[l] + portion_length
        dist_vectors.append(node_dist_vec)

        node_partition_result.append(node_dataset)


    final_list = []
    for item in node_partition_result:
        x_list = []
        y_list = []
        for j in item:
            x_list.append(j[0])
            y_list.append(j[1])

        final_list.append((x_list,y_list))
    return (num_nodes,final_list,intervals,dist_vectors)


def random_sample(x_data,y_data, batch_size):
    data_index = random.sample(range(len(x_data)), batch_size)
    x = []
    y = []
    for i in data_index:
        x.append(x_data[i])
        y.append(y_data[i])
    return np.array(x),np.array(y)

def calculate_intervals(partition_list,quantity_dict,lmda = 1,min_interval=1):
    D = 0
    for key in quantity_dict:
        D += quantity_dict[key]

    index_list = []
    for label_list in partition_list:
        D_i = 0
        local_quantity_dict = {}
        for label, portion in enumerate(label_list):
            local_quantity_dict[label] = int((portion * quantity_dict[label])/100.0)
            D_i += local_quantity_dict[label]

        quality_index = 0
        for label in quantity_dict.keys():
            p_l = float(local_quantity_dict[label]) / D_i
            P_l = float(quantity_dict[label]) / D
            p_prime = float(quantity_dict[label] - local_quantity_dict[label]) / quantity_dict[label]
            quality_index += (2*p_l*p_prime + abs(p_l - P_l))*lmda

        index_list.append(quality_index)

    index_list = np.array(index_list) / np.min(index_list)

    interval_list = np.ceil(index_list * min_interval)
    return interval_list

def data_aug(x_train):
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        )

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    return datagen
