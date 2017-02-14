import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import config
import process_data

def plot_signal(img,axis):
    """Simply plot a sample image from dataset"""
    axis.imshow(img)
    axis.set_xticks([])
    axis.set_yticks([])
    return axis


############################################################
# Configuration:
############################################################
input_files = '/home/carnd/Behavioral_Cloning/initial_files/'

x_pix = config.x_pix
y_pix = config.y_pix
seed = 2016
test_size = 0.2

# for reproducibility
np.random.seed(seed)

############################################################
# Load and process Data_test:
############################################################
with open(input_files + 'features_{0}.pickle'.format(config.version), 'rb') as handle:
    X = pickle.load(handle)
with open(input_files + 'labels_{0}.pickle'.format(config.version), 'rb') as handle:
    y = pickle.load(handle)

X = X.astype('float32')
y = y.astype('float32')

print(X.shape)

# the histogram of y labels:
n, bins, patches = plt.hist(y, 100, facecolor='#03B3E4', alpha=0.75)

plt.xlabel('Steering Angles')
plt.ylabel('Num of Samples')
plt.title('Steering Angles distributions')
plt.grid(True)

plt.savefig(input_files + 'y_labels_distr_balanced' + '.png')


# Visualize a sequence of images:
idx = 350
sequence = X[idx]
sequence_label = y[idx]



# Apply augmentation over sequence of data:


cv2.imwrite(input_files + 'example.png', sequence)
# random value:
intensity = np.random.uniform()
intensity = 0.7
# random flipping:
flipping = np.random.choice([True, False])
flipping = True

img, y_steer = process_data.augmented_images(sequence, sequence_label, flipping, intensity)
cv2.imwrite(input_files + 'example_aug.png', img)
print(sequence_label, y_steer)





f, axarr = plt.subplots(2, 2, figsize=(10, 10))
augmented_angles = []
for i in range(0, X.shape[0]*100):
    # random value:
    intensity = np.random.uniform()
    # random flipping:
    flipping = np.random.choice([True, False])
    # random sample
    idx = np.random.randint(X.shape[0])
    _, steering_aug = process_data.augmented_images(X[idx], y[idx], flipping, intensity)
    augmented_angles.append(steering_aug)
    if i in [1000, 3000, 5000, 7000]:
        print('Printing....')
        # the histogram of y labels:
        if i == 1000:
            n, bins, patches = axarr[0, 0].hist(augmented_angles, 100, facecolor='#81F7F3', alpha=0.75)
            axarr[0, 0].set_title('Distribution after {0} iterations'.format(i))
        if i == 3000:
            n, bins, patches = axarr[0, 1].hist(augmented_angles, 100, facecolor='#2E64FE', alpha=0.75)
            axarr[0, 1].set_title('Distribution after {0} iterations'.format(i))
        if i == 5000:
            n, bins, patches = axarr[1, 0].hist(augmented_angles, 100, facecolor='#642EFE', alpha=0.75)
            axarr[1, 0].set_title('Distribution after {0} iterations'.format(i))
        if i == 7000:
            n, bins, patches = axarr[1, 1].hist(augmented_angles, 100, facecolor='#03B3E4', alpha=0.75)
            axarr[1, 1].set_title('Distribution after {0} iterations'.format(i))

f.suptitle('Augmented Data', fontsize=14, fontweight='bold')
f.savefig(input_files + 'Augmented_distribution.png'.format(idx))









