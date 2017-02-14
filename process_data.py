import pandas
import numpy as np
import cv2
import pickle

import config


def brightness_images(image, intensity):
    """
    Function to modify image bright
    :param image: image array
    :param intensity: random value between 0 and 1
    :return: image array modified
    """
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = np.clip(intensity, 0.1, 0.9)
    image1[:, :, 2] = image1[:, :, 2]*random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def trans_image(image, steer, trans_range, intensity):
    """
    Function to translate image in both axis
    :param image: image array
    :param steer: steeering angle associated
    :param trans_range: maximum number of pixels to translate
    :param intensity: random value between 0 and 1
    :return: image array modified
    """
    # Translation
    tr_x = trans_range * intensity - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = (trans_range * 0.4) * intensity - (trans_range * 0.4) / 2
    trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    rows, cols = image.shape[:2]
    image_tr = cv2.warpAffine(image, trans_M, (cols, rows))
    return image_tr, steer_ang


def augmented_images(img, steering_angle, flipping, intensity):
    """
    Generates augmented pictures changing brightness, flipping or translation
    :param img: image array
    :param steering_angle: corresponding steering angle
    :param intensity: numpy random transformation
    :param flipping: Boolean flipping
    :return: augmented image and steering angle
    """
    # brightness image:
    img = brightness_images(img, intensity)
    # flip image:
    if flipping:
        img = cv2.flip(img, 1)
        steering_angle *= -1.
    # translate image:
    img, y_steer = trans_image(img, steering_angle, config.max_translation, intensity)
    return img, y_steer


def process_images(path_img, x_pix, y_pix, y_crop):
    """
    Resize and Normalize images
    :param path_img: image path
    :param x_pix: final x pixels size
    :param y_pix: final y pixels size
    :param y_crop: y pixels to crop image
    :return: processed image
    """
    # read image:
    img = cv2.imread(path_img, 1)
    # normalize image:
    cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # crop image:
    img = img[y_crop:, :, :]
    # resize image:
    img = cv2.resize(img, (x_pix, y_pix), interpolation=cv2.INTER_CUBIC)
    return img


############################################################
# Configuration parameters:
############################################################

path_to_log = '../initial_files/Data/driving_log.csv'

current_path = '/home/carnd/Behavioral_Cloning/initial_files/'

############################################################
# Process Data_test:
############################################################
if __name__ == "__main__":
    print("Processing...")

    df = pandas.read_csv(path_to_log, sep=",")

    # some required transformation in the dataset:
    df["center"] = df['center'].apply(lambda x: "../initial_files/Data/" + x)
    df[['steering', 'throttle', 'brake', 'speed']] = df[['steering', 'throttle', 'brake', 'speed']].astype(float)

    # Generate a list of tuples where the first element is the numpy array with images and the second element the
    # corresponding steering angle:
    dataset = [
        (process_images(df.loc[idx, 'center'], config.x_pix, config.y_pix, config.y_crop), df.loc[idx, 'steering']) for
        idx in range(0, df.shape[0])]

    # remove straight drives:
    dataset = [tup for tup in dataset if tup[1] != 0]

    # Generate X and Y sets for model_CNN training:
    X = np.concatenate([x[0][np.newaxis, :] for x in dataset], axis=0)
    Y = np.asarray([x[1] for x in dataset])

    # identify straight drives and remove 90% of them:
    straight_drives = np.where(np.abs(Y) < 0.1)[0]
    num_sd = len(straight_drives)
    print('Straight drives removed: ', int(0.99*num_sd))
    X = np.delete(X, straight_drives[:int(0.99*num_sd)], axis=0)
    Y = np.delete(Y, straight_drives[:int(0.99*num_sd)])

    print("Dataset shape:", X.shape)

    # Save files to pickle:
    with open(current_path + 'features_{0}.pickle'.format(config.version), 'wb') as handle:
        pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(current_path + 'labels_{0}.pickle'.format(config.version), 'wb') as handle:
        pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Process completed!")
