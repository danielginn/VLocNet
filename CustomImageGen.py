import keras.backend as K
import tensorflow_probability as tfp
import numpy as np
from keras.preprocessing.image import load_img
from keras.callbacks import Callback
from keras.metrics import Metric
from keras.preprocessing.image import img_to_array
from scipy.spatial.transform import Rotation as R
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow as tf
import glob
import json
import math
import random


class MyMetrics(Callback):
    def __init__(self, val_data, batches, batch_size=32):
        self.validation_data = val_data
        self.batch_size = batch_size
        self.batches = batches

    def on_test_begin(self, logs=None):
        self.errors = []

    #def on_epoch_end(self, epoch, logs=None):
    def on_test_end(self, logs=None):
        total =  self.batches * self.batch_size
        val_pred = np.zeros((total,7))
        val_true = np.zeros((total,7))

        for batch in range(self.batches):
            xVal, yVal = next(self.validation_data)
            val_pred[batch * self.batch_size : (batch+1) * self.batch_size, :] = np.asarray(self.model.predict(xVal))
            val_true[batch * self.batch_size : (batch+1) * self.batch_size, :] = yVal

        x_diff = val_true[:, 0] - val_pred[:, 0]
        y_diff = val_true[:, 1] - val_pred[:, 1]
        z_diff = val_true[:, 2] - val_pred[:, 2]
        xyz_error = np.sqrt(np.square(x_diff)+np.square(y_diff)+np.square(z_diff))

        for e in xyz_error:
            self.errors.append(e)

    def get_median(self):
        self.median = np.median(self.errors)
        return self.median

    def get_outliers(self):
        std = np.std(self.errors)
        outliers = []
        i = 0
        for e in self.errors:
            i += 1
            if e > (self.median + 4*std):
                outliers.append((e, i))

        return outliers

    def get_all_errors(self):
        return self.errors






def geo_loss(y_true, y_pred):
    x_diff = y_true[:, 0] - y_pred[:, 0]
    y_diff = y_true[:, 1] - y_pred[:, 1]
    z_diff = y_true[:, 2] - y_pred[:, 2]

    #q_pred = K.l2_normalize(y_pred[:, 3:7], axis=1)
    q1_diff = y_true[:, 3] - y_pred[:, 3]
    q2_diff = y_true[:, 4] - y_pred[:, 4]
    q3_diff = y_true[:, 5] - y_pred[:, 5]
    q4_diff = y_true[:, 6] - y_pred[:, 6]

    L_x = K.sqrt(K.square(x_diff) + K.square(y_diff) + K.square(z_diff))
    L_q = K.sqrt(K.square(q1_diff) + K.square(q2_diff) + K.square(q3_diff) + K.square(q4_diff))

    B = 1
    return L_x + B*L_q

def xyz_error(y_true, y_pred):
    x_diff = y_true[:, 0] - y_pred[:, 0]
    y_diff = y_true[:, 1] - y_pred[:, 1]
    z_diff = y_true[:, 2] - y_pred[:, 2]
    xyz_error = K.sqrt(K.square(x_diff) + K.square(y_diff) + K.square(z_diff))

    median_error = tfp.stats.percentile(xyz_error, q=50, interpolation='midpoint')

    return median_error


def q_error(y_true, y_pred):
    return tf_function(y_true, y_pred)


@tf.function(input_signature=[tf.TensorSpec(shape=[4, 4], dtype=tf.float32), tf.TensorSpec(shape=[4, 4], dtype=tf.float32)])
def tf_function(input1, input2):
    y = tf.numpy_function(quat_diff, [input1, input2], tf.float32)
    return y


def quat_diff(y_true, y_pred):
    R_true = R.from_quat(y_true)
    R_pred = R.from_quat(y_pred)
    R_diff = R_true.inv()*R_pred
    q_diff = R_diff.as_quat()

    lengths = np.sqrt(np.square(q_diff[:, 0]) + np.square(q_diff[:, 1]) + np.square(q_diff[:, 2]))
    angles = np.degrees(2 * np.arctan2(lengths, q_diff[:, 3]))
    for i in range(angles.shape[0]):
        if angles[i] > 180:
            angles[i] = 360 - angles[i]
    median_error = np.mean(angles)
    #median_error = tfp.stats.percentile(angles, q=50, interpolation='midpoint')
    return K.cast(median_error, dtype='float32')

def list_of_folders(purpose,exclude):
    list = []
    if purpose == "train":
        for i in range(49):
            if i+1 != exclude:
                list.append("./360cameraNUbotsField/dataset4/" + str(i+1).zfill(2) + "/")

    else:
        list.append("./360cameraNUbotsField/dataset4/" + str(exclude).zfill(2) + "/")
    return list

def list_of_folders2(purpose):
    # Train on 40
    list = []
    if purpose == "train":
        i = 1
        for j in range(3):
            for k in range(7):
                list.append("./360cameraNUbotsField/dataset4/" + str(i).zfill(2) + "/")
                i += 1

            for k in range(4):
                list.append("./360cameraNUbotsField/dataset4/" + str(i).zfill(2) + "/")
                i += 2
            i -= 1

        for k in range(7):
            list.append("./360cameraNUbotsField/dataset4/" + str(i).zfill(2) + "/")
            i += 1
    else:
        i = 9
        for j in range(3):
            for k in range(3):
                list.append("./360cameraNUbotsField/dataset4/" + str(i).zfill(2) + "/")
                i += 2
            i += 8
    return list

def list_of_folders3(purpose):
    #Train on 45
    list = []
    if purpose == "train":
        for i in range(49):
            if (i+1 != 1) and (i+1 != 7) and (i+1 != 43) and (i+1 != 49):
                list.append("./360cameraNUbotsField/dataset4/" + str(i + 1).zfill(2) + "/")
    else:
        list.append("./360cameraNUbotsField/dataset4/01/")
        list.append("./360cameraNUbotsField/dataset4/07/")
        list.append("./360cameraNUbotsField/dataset4/43/")
        list.append("./360cameraNUbotsField/dataset4/49/")

    return list

def list_of_folders_from_file(file):
    with open(file) as f:
        lines = f.read().splitlines()
    return lines

def list_of_files(dataset, purpose, folders):
    if dataset == "NUbots":
        return [f for f in glob.glob("./NUbotsField/"+purpose+"/*/*.jpg")]
    elif dataset == "7scenes":
        return [f for f in glob.glob("./7scenes/*/" + purpose + "/*/*.color.png")]
    elif dataset == "360cameraNUbotsField":
        list = []
        for f in folders:
            for i in range(1799): # Why was 719 here?
                list.append(f + str(i).zfill(4) + ".JPG")
        return list
    elif dataset == "BlenderNUbotsField":
        list = []
        for f in folders:
            for i in range(1229):  # Why was 719 here?
                list.append(f + str(i) + ".png")
        return list
    elif dataset == "BlenderRoboCup":
        list = []
        for f in folders:
            for file in glob.glob(f+"/*.png"):
                list.append(file)
        return list
    else:
        return []



def center_crop(img, crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = crop_size
    x = (width-dx)//2 + 1
    y = (height-dy)//2 + 1
    return img[y:(y+dy), x:(x+dx), :]


def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


def crop_generator(image, crop_length, isRandom):
    if isRandom:
        image_crop = random_crop(image, (crop_length, crop_length))
    else:
        image_crop = center_crop(image, (crop_length, crop_length))
    return image_crop


def get_input(path,is_noise):
    img_full = load_img(path)

    if np.char.startswith(path, "./7scenes"):
        img_resized = img_full.resize((341, 256), Image.ANTIALIAS)
        img_np = img_to_array(img_resized)
        cropped_image = crop_generator(img_np, 224, isRandom=True)
    elif np.char.startswith(path, "./NUbotsField"):
        #scale = 1
        height = 224# / scale
        img_resized = img_full.resize((height,height), Image.ANTIALIAS)
        img_np = img_to_array(img_resized)
        cropped_image = img_np
        #cropped_image = crop_generator(img_np, 224, isRandom=False)
        if is_noise:
            cropped_image = add_double_noise(cropped_image)

    else:
        cropped_image = img_to_array(img_full)
        #if is_noise:
            #cropped_image = add_noise(cropped_image)

    return cropped_image

def add_noise(image):
    obscure_num = random.randint(0,3)
    if (obscure_num > 0):
        obscure_pcnt = obscure_num * 0.1
        long_dim = round(math.sqrt(50176 * obscure_pcnt) * 1.414)
        short_dim = round(long_dim / 2)
        rand_hor = random.randint(0, 224 - short_dim)
        rand_ver = 112-short_dim
        for m in range(rand_ver, rand_ver + long_dim):
            for n in range(rand_hor, rand_hor + short_dim):
                for c in range(3):
                    image[m, n, c] = 0
    return image

def add_double_noise(image):
    obscure_num = random.randint(0, 3)
    if (obscure_num > 0):
        obscure_pcnt = obscure_num * 0.05
        long_dim = round(math.sqrt(50176 * obscure_pcnt) * 1.414)
        short_dim = round(long_dim / 2)
        rand_hor_left = random.randint(0, 112 - short_dim)
        rand_hor_right = random.randint(112, 224 - short_dim)
        rand_ver = 112 - short_dim
        for m in range(rand_ver, rand_ver + long_dim):
            for c in range(3):
                for n in range(rand_hor_left, rand_hor_left + short_dim):
                    image[m, n, c] = 0
                for n in range(rand_hor_right, rand_hor_right + short_dim):
                    image[m, n, c] = 0
    return image

def get_inputs(path,is_noise):
    img_full = load_img(path)
    img_resized = img_full.resize((341, 256), Image.ANTIALIAS)
    img_np = img_to_array(img_resized)
    cropped_image = crop_generator(img_np, 224, isRandom=True)
    if is_noise:
        cropped_image = add_noise(cropped_image)

    prev_num = int(path[-10]) - 1
    pose_path = path[:-11] + str(prev_num) + path[-9:]
    print(path)
    print(pose_path)
    print("-----------------")
    xyzq = get_output(pose_path)

    return cropped_image, xyzq


def get_output(image_path):
    if np.char.startswith(image_path, "./7scenes"):
        xyzq = np.zeros(7)
        pose_path = image_path[:-9] + "pose.txt"
        file_handle = open(pose_path, 'r')

        # Read in all the lines of your file into a list of lines
        lines_list = file_handle.readlines()
        # Do a double-nested list comprehension to store as a Homogeneous Transform matrix
        homogeneous_transform_list = [[float(val) for val in line.split()] for line in lines_list[0:]]
        homogeneous_transform = np.zeros((4, 4))

        for j in range(4):
            homogeneous_transform[j, :] = homogeneous_transform_list[j]

        # Extract xyz from homogeneous Transform
        xyzq[0:3] = homogeneous_transform[0:3, 3]
        # Extract rotation from homogeneous Transform
        r = R.from_dcm(homogeneous_transform[0:3, 0:3])
        xyzq[3:7] = r.as_quat()

        file_handle.close()
        return xyzq

    elif np.char.startswith(image_path, "./NUbotsField"):
        json_filename = image_path[:-3] + "json"
        xyzq = np.zeros(7)

        with open(json_filename) as f2:
            json_data = json.load(f2)
        xyzq[0:3] = json_data['position']
        xyzq[3:7] = json_data['rotation']
        #print(xyzq)
        return xyzq

    else:
        json_filename = image_path[:-3] + "json"
        xyzq = np.zeros(7)
        with open(json_filename) as f2:
            json_data = json.load(f2)
        xyzq[0:2] = json_data['position'][0:2]
        xyzq[2] = 0.831
        angle = json_data['orientation']
        r = R.from_euler('z',angle,degrees=True)
        xyzq[3:7] = r.as_quat()
        return xyzq

    return 0


def image_generator(files, batch_size, is_random, is_noise):
    i = 0



    while True:
        # Select files (paths/indices) for the batch
        if is_random:
            batch_paths = np.random.choice(files, batch_size)
        else:
            batch_start = max(((i + batch_size) % len(files)) - batch_size, 0)  # This makes sure a batch too close to the end is not chosen
            batch_end = batch_start + batch_size
            i += batch_size
            batch_paths = np.array(files[batch_start:batch_end])


        batch_input = []
        batch_output = []
        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            input = get_input(input_path, is_noise)
            output = get_output(input_path)

            batch_input += [input]
            batch_output += [output]

        # Return a tuple of (input, output) to feed the network
        datagen = ImageDataGenerator(rotation_range=10)

        batch_x = datagen.standardize(np.array(batch_input))
        batch_y = np.array(batch_output)
        yield (batch_x, batch_y)