#import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import string
import re

def load_image(filename, image_size = (224,224)):
	'''
	FUNCIONALITIES: read a image, resize it to specific size, and return a tf tensor
	ARGUMENTS:
	- filename: full file path or url -> string
	- image_size: desired size of image (height, width) -> tuple of int
	RETURN:
	- a tf tensor of shape (height, width, channels) like (224,224,3)
	USAGE:
	img = load_image(
		'/home/project/classification/pet/cat/1.jpg', # file name
		image_size = (180,180) # size that you want to resize to
	)
	'''
	# read a file from full file path or url
    img = tf.io.read_file(filename)
	# decode the image into a tensor
    img = tf.image.decode_jpeg(img)
	# resize it to image_size
    img = tf.image.resize(img, image_size)
    return img


def pred_and_plot(img, model, class_names):
    '''
	FUNCTIONALITIES: predict and plot a image
	ARGUMENTS:
    - img: array-like of shape (height, width, 3) -> numpy array or tf tensor
    - model: keras trained model
    - class_names: name of classes -> list of string
	RETURN: NONE
	USAGE:
	pred_and_plot(img, catdog_model, ['cat', 'dog'])
    '''
	# expand dimension to (batch, height, width, 3)
    img = tf.expand_dims(img, axis = 0)
	# predict
    pred = model.predict(img)
	# get the label from 2D one hot encoding array
    pred_label = np.argmax(pred, axis = 1)[0]
	# plot the image
    plt.imshow(img[0].numpy().astype('uint8'))
    plt.title(f'predict: {class_names[pred_label]}')
    plt.axis('off')
    plt.show()
	
	
def class_distribution(y):
    '''
	FUNCTIONALITIES: plot class distribution
	ARGUMENTS:
    - y: labels -> numpy array or tf dataset
	RETURN: NONE
	USAGE:
	class_distribution(y_train)
	class_distribution(train_dataset)
    '''
    # if y is numpy array
    if isinstance(y, np.ndarray):
        labels = y
    # if y is tf dataset
    else:
		# create a numpy array from a list of labels (numpy arrays)
        labels = np.array(
            list(
                y.unbatch().map(lambda img, lbl: lbl).as_numpy_iterator() # return a list of labels (numpy form)
            )
        )
	# if labels is in 2D one hot encoding form -> use argmax to get labels in 1D form (like [0,3,2,1,2,3])
    if labels.ndim == 2:
        n_classes = labels.shape[1] # number of class n_classes = number of features
        labels = np.argmax(labels, axis = 1)
    else:
        n_classes = len(
            np.unique(labels) # number of class n_classes = number of unique values in labels, e.g [1,1,0,2] -> n_classes = 3
        )
	# plot histogram of the labels
    plt.hist(labels, bins = n_classes, rwidth = 0.8)
    plt.show()


def visualize_image_data(X, y = None,
                          class_names = None,
                          aug = None,
                          label_key = None,
                          pred_probs = None,
                          figsize = (10,12),
                          num_images = 25,
                          cmap = None):
    '''
	FUNCTIONALITIES: visualize image data from numpy array or tf dataset
	ARGUMENTS:
    - X: images data -> numpy array or tf dataset
    - y: labels -> nump array or None (if X is a tf dataset)
    - class_names: name of classes -> list of string (e.g [0,1,2] or ['cat', 'dog', 'monkey']) or None
    - aug: data augmentation -> keras Sequential or None
    - label_key: visualize a specific class -> int, string or None 
    - pred_probs: prediction probabilities -> numpy array in one hot encoding form
    - figsize: figure size -> tuple of int
    - num_images: number of images in figure -> int
    - cmap: color map -> plt.cm.binary (to visualize grey image) or None (color image) or 'grey'
	RETURNS: NONE
	USAGE:
	- visualize training data
	visualize(train_dataset)
	visualize(x_train, y_train, class_names = ['cat', 'dog', 'monkey'])
	- visualize testing data
	visualize(test_dataset, pred_probs = pred_probs)
    '''
	# PRODUCE images, true_labels, class_names FROM X, y, class_names
    # if X is numpy array -> assign simply
    if isinstance(X, np.ndarray):
        images = X
        true_labels = y
    #if X is tf dataset
    else:
        # if we want to visualize prediction (on testing set), we will take all images
        # otherwise, we will take 1 batch for simplicity
        if class_names is None: # if we don't provide class_names -> we will get it from tf dataset
			class_names = X.class_names # maybe raise an error (if inputs are not numpy array or tf dataset or tf dataset doesn't have labels)
        if label_key is None and pred_probs is None: # if we don't provide label_key, pred_probs together -> visualize training data
            ds = X.take(1).unbatch() # take 1 batch
        else:# otherwise -> visualize prediction (on testing data)
            ds = X.unbatch() # take all images
        # if tf dataset spec is tuple -> it includes (images, labels)
        # otherwise -> the tf dataset includes only images
        if isinstance(X.element_spec, tuple):
            # tf dataset can be shuffled when using -> use for loop, not .as_numpy_iterator()
            images = []
            true_labels = []
            for img, lbl in ds:
                images.append(img)
                true_labels.append(lbl)
            images = np.asarray(images)
            true_labels = np.asarray(true_labels)
        else:
            images = np.array(
                list(
                    ds.as_numpy_iterator()
                )
            )
            true_labels = None
    # if true_labels is None -> create class_names = [None]
    # and true_labels = [0,0,...,0] to represent label = None for all images
    if true_labels is not None:
        if true_labels.ndim == 2:
            if class_names is None:
                class_names = np.array(range(true_labels.shape[1]))
            true_labels = np.argmax(true_labels, axis = 1)
        else:
            if class_names is None:
                class_names = np.array(
                    range(
                        len(np.unique(true_labels))
                    )
                )
    else:
        class_names = [None]
        true_labels = np.array([0]*len(images))

    # PRODUCE label_key, pred_labels FROM label_keys
    if isinstance(label_key, str):
        if label_key not in class_names:
            print('provided "label_key" not in "class_names"!')
            return
        label_key = class_names.index(label_key)
    # PRODUCE pred_labels, prob_labels FROM pred_probs
	if pred_probs is not None:
        # predicted label
        pred_labels = np.argmax(pred_probs, axis = 1)
        # probability of label prediction
        prob_labels = np.max(pred_probs, axis = 1)*100

    # VISUALIZE
    plt.figure(figsize = figsize)
    if cmap == 'grey':
        cmap = plt.cm.binary
    n_samples = len(images)
    r = math.ceil(math.sqrt(num_images))# plot r images x r images on figure
    # shuffle the images
    shuffled_indices = np.random.permutation(n_samples)
    idx0 = shuffled_indices[0] # use only 1 image in case of visualizing augmentation data
    # visualize testing data
	if label_key is not None:
        j = 0 # images counter
        for i in range(n_samples):
            idx = shuffled_indices[i] # index that shuffled
            if true_labels[idx] == label_key: # if true label is label key -> visualize it
                j += 1
                if j > min(num_images, n_samples): break
                plt.subplot(r,r,j)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(images[idx].astype('uint8'), cmap = cmap)
                plt.title(f'label: {class_names[true_labels[idx]]}') # show true label on top
                if pred_probs is not None:
                    plt.xlabel( # show predicted label on bottom
                        f'prediction: {class_names[pred_labels[idx]]}' +
                        ' ({:2.2f}%)'.format(prob_labels[idx])
                    )
        plt.show()
        return 
    # visualize training data
	for i in range(min(num_images,n_samples)):
        plt.subplot(r,r,i+1)
        idx = shuffled_indices[i]
        plt.xticks([])
        plt.yticks([])
        if aug is None:
            plt.imshow(images[idx].astype('uint8'), cmap = cmap)
            plt.title(f'label: {class_names[true_labels[idx]]}')
            if pred_probs is not None:
                plt.xlabel(
                    f'prediction: {class_names[pred_labels[idx]]}' +
                    ' ({:2.2f}%)'.format(prob_labels[idx])
                )
        else:
            plt.imshow(aug(images[idx0]).numpy().astype('uint8'), cmap = cmap)
            plt.title(f'label: {class_names[true_labels[idx0]]}')
    plt.show()
	
	
	
	
	
	
	
