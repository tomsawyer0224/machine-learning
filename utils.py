#import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import string
import re
import itertools
from sklearn.metrics import classification_report, confusion_matrix
import datetime

# IMAGE CLASSIFICATION UTILITIES
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
        filename = '/home/project/classification/pets/cat/1.jpg', # file name
        image_size = desired_image_size # size that you want to resize to
    )
    img = load_image(
        filename = 'https://storage/car.jpg',
        image_size = desired_image_size
    )
    '''
    # read a file from full file path or url
    img = tf.io.read_file(filename)
    # decode the image into a tensor
    img = tf.image.decode_jpeg(img)
    # resize it to image_size
    img = tf.image.resize(img, image_size)
    return img

def predict_and_plot(image, image_size, model, class_names, cmap = None):
    '''
    FUNCTIONALITIES: 
    - predict an image
    - plot result
    ARGUMENTS:
    - image: an image -> numpy array, tf tensor of shape (height, width, channels), file name, url
    - image_size: image will be resized to image_size (without channels)
    - model: keras classification model
    - class_names: name of classes -> list
    - cmap: color map -> plt.cm.binary or 'grey' or None
    RETURN: None
    USAGE:
    predict_and_plot(image=3D_array, model=cat_dog_model, class_names=['cat', 'dog'])
    predict_and_plot(image='pets/cats/cat1.jpg', model=cat_dog_model, class_names=['cat', 'dog'])
    predict_and_plot(image='http://storage/dog2.jpg', model=cat_dog_model, class_name=['cat', 'dog'])
    '''
    # get image from file name or url
    if isinstance(image, str):# -> file name or url 
        plt.title(image)
        image = load_image(image, image_size = image_size)
    # else -> do nothing
    img = tf.expand_dims(image, axis = 0)# expand to 4D array
    pred = model.predict(img)# predict
    if pred.shape[1] == 1: # shape of pred is (None, 1) -> binary classification
        pred_label = 1 if pred[0][0] > 0.5 else 0 # get label
        prob_label = pred[0][0] if pred[0][0] > 0.5 else 1 - pred[0][0] # get probability of predicted label
    else:# categorical classification
        pred_label = np.argmax(pred, axis = 1)[0] # get label
        prob_label = np.max(pred, axis = 1)[0] # get probability of predicted label
    if cmap == 'grey':
        cmap = plt.cm.binary
    plt.imshow(img[0].numpy().astype('uint8'), cmap = cmap)# show image
    plt.xlabel(f'predict: {class_names[pred_label]}' + ' ({:2.2f}%)'.format(100*prob_label))
    plt.xticks([])
    plt.yticks([])
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
        if n_classes == 1:
            labels = labels.flatten().astype('int32')
        else:
            labels = np.argmax(labels, axis = 1)
    else:
        n_classes = len(
            np.unique(labels) # number of class n_classes = number of unique values in labels, e.g [1,1,0,2] -> n_classes = 3
        )
    # plot histogram of the labels
    plt.hist(labels, bins = n_classes, rwidth = 0.8)
    plt.show()

def visualize_image_data(
    X, 
    y = None,
    class_names = None,
    aug = None,
    label_key = None,
    pred_probs = None,
    figsize = (15,18),
    num_images = 25,
    cmap = None
):
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
    - visualize on training data
    visualize_image_data(train_dataset)
    visualize_image_data(x_train, y_train, class_names = ['cat', 'dog', 'monkey'])
    - visualize on testing data
    visualize_image_data(x_test, y_test, class_names = ['cat', 'dog', 'monkey'])
    visualize_image_data(x_test, y_test, class_names = ['cat', 'dog', 'monkey'], pred_probs = pred_probs ,label_key = 'cats')
    visualize_image_data(test_dataset, pred_probs = pred_probs)
    visualize_image_data(test_dataset, pred_probs = pred_probs, label_key = 1)
    '''
    # FIGURE CONFIGURATIONS
    if cmap == 'grey':
        cmap = plt.cm.binary # for grey images
    plt.figure(figsize = figsize)
    r = math.ceil(math.sqrt(num_images))# number of images on width, height dimensions
    
    # VISUALIZE AUGMENTED IMAGES
    if aug:
        if isinstance(X, np.ndarray): # if X is numpy array
            img = X[0]
        else: # X is tf dataset
            element_spec = X.element_spec
            elem = next(iter(X.unbatch()))
            if isinstance(element_spec, tuple): # each element is tuple (image, label)
                img = elem[0]
            else: # each element is image (without label)
                img = elem
        # plot images
        for i in range(num_images):
            plt.subplot(r,r,i+1)
            plt.imshow(aug(img).numpy().astype('uint8'), cmap = cmap)
            plt.axis('off')
        return
    
    # IF INPUTS ARE NUMPY ARRAY -> CREATE TF DATASET FROM THEM
    if isinstance(X, np.ndarray):
        if class_names is None:
            print('please provide "class_names"')
            return
        if y is None:
            y = np.array([-1]*len(X), dtype = 'int32')
        X = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)
    
    # GET CLASS NAMES
    if class_names is None:
        class_names = X.class_names
    
    # UNBATCH AND MAP THE DATASET:
    # e.g batched dataset (images - 4D tensor, labels - 2D tensor, 1D tensor)
    # is converted unbatched dataset (image - 3D tensor, label - 0D tensor)
    element_spec = X.element_spec # element spec of original dataset (batched dataset)
    if isinstance(element_spec, tuple): # element = (images, labels)
        label_spec = element_spec[1] # label spec
        if len(label_spec.shape) == 2: # labels in categorical or binary mode
            if label_spec.shape[1] == 1: # binary mode
                #ds = X.unbatch().map(lambda img, lbl: (img, int(lbl[0])))    
                ds = X.unbatch().map(lambda img, lbl: (img, tf.cast(lbl[0], dtype = tf.int32)))
            else: # categorical mode
                ds = X.unbatch().map(
                    lambda img, lbl: (img, tf.math.argmax(lbl, output_type = 'int32'))
                )     
            # NOTE: after unbatching, the labels in 2D form is changed to 1D form 
        else:# labels in int mode
            ds = X.unbatch()
    else: # element = image -> convert to element = (images, label = -1)
        ds = X.unbatch().map(lambda img: (img, -1))
    # ENUMERATE FOR INDEXING PURPOSE 
    ds = ds.enumerate()
    
    # EXPAND THE DATASET IF PROVIDE PREDICTED PROBABILITIES
    # pred_probs is 2D numpy array
    if pred_probs is not None:
        if pred_probs.shape[1] == 1: # shape (None, 1) -> binary classification
            pred_probs_flattenned = pred_probs.flatten()
            pred_labels = np.where(pred_probs_flattenned > 0.5, 1, 0)
            prob_labels = 100*np.where(pred_probs_flattenned > 0.5, pred_probs_flattenned, 1 - pred_probs_flattenned)
        else:
            # predicted label
            pred_labels = np.argmax(pred_probs, axis = 1)
            # probability of predicted label
            prob_labels = 100*np.max(pred_probs, axis = 1)
        # convert to tensor
        pred_labels = tf.constant(pred_labels, dtype = 'int32')
        prob_labels = tf.constant(prob_labels, dtype = 'float32')
        
        ds = ds.map(
            lambda i, elem: (i, elem + (pred_labels[i], prob_labels[i]))
        )
    else:
        ds = ds.take(num_images)
    
    # GET LABEL KEY IF PROVIDED
    if isinstance(label_key, str):
        if label_key not in class_names:
            print('provided "label_key" not in "class_names"!')
            return
        label_key = class_names.index(label_key)
    
    # FILTER THE DATASET IF LABEL KEY IS PROVIDED
    if label_key is not None:
        ds = ds.filter(lambda i, elem: tf.math.equal(elem[1], label_key))
    ds = ds.map(lambda i, elem: elem)
    
    # COUNT NUMBER OF SAMPLES IN THE DATASET
    n_samples = 0
    for _ in ds:
        n_samples += 1
    # EXTEND CLASS NAME (THE LAST ELEMENT IS '?') FOR VISUALIZING UNKNOW LABELS
    class_names_extended = class_names + ['?']

    shuffled_indices = np.random.permutation(n_samples) # shuffle the indices
    idx = shuffled_indices[:num_images] # get 'num_images' images from the dataset
    j = 0 # image counter
    for i, elem in enumerate(ds):
        if i in idx:
            j += 1
            if j > num_images: return
            img = elem[0]
            lbl = elem[1]

            plt.subplot(r,r,j)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.numpy().astype('uint8'), cmap = cmap)
            plt.title(f'label: {class_names_extended[lbl]}')
            if pred_probs is not None:
                pred = elem[2]
                prob = elem[3]
                plt.xlabel(f'prediction: {class_names_extended[pred]} ({prob:2.2f}%)')
    return

def prediction_result(X, y = None,
                      class_names = None,
                      filenames = None,
                      pred_probs = None,
                      return_wrong_pred_only = True,
                      figsize = (15,18),
                      cmap = None):
    '''
    FUNCTIONALITIES: 
    - creat a pandas DataFrame to describe prediction result
    - plot top wrong predictions
    ARGUMENTS:      
    - X: images -> numpy array or tf dataset without shuffle
    - y: lables -> numpy array (1D or 2D) or None (if X is tf dataset)
    - class_names: name of classes -> list
    - filenames: None (if X is numpy array) or list of file names (if X is dataset)
        Note: file name must correspond to (image, label) in ds
    - pred_probs: prediction probabilities -> 2D numpy array
    - return_wrong_pred: return only wrong predictions (True) or return all predictions (False) -> boolean
    ------2 args below for plotting purpose
    - figsize: figure size -> tuple of int
    - cmap: color map -> plt.cm.binary or None or 'grey'
        
    RETURN: a pandas DataFrame
    USAGE:
    - numpy array: 
        pred_result = prediction_result(X=X_test, y=y_test, class_names=class_names, pred_probs=test_pred_probs)
        if we don't provide class_names -> class_names = class id
    - tf dataset: 
        pred_result = prediction_result(X=test_dataset, pred_probs=test_pred_probs, filenames=list_file_names_in_test_dir)
    '''
    # PRODUCE pred_labels, prob_labels FROM pred_probs
    if pred_probs is None:
        print('please provide "pred_probs"!')
        return
    
    if pred_probs.shape[1] == 1: # (None, 1) -> binary classification
        pred_probs_flattended = pred_probs.flatten()
        pred_labels = np.where(pred_probs_flattended > 0.5, 1, 0)
        prob_labels = np.where(pred_probs_flattended > 0.5, pred_probs_flattended, 1 - pred_probs_flattended)
    else: # categorical classification
        # predicted label
        pred_labels = np.argmax(pred_probs, axis = 1)
        # probability of prediction
        prob_labels = np.max(pred_probs, axis = 1)
    # ------------------------------------------------
    
    # CREATE TF DATASET FROM NUMPY ARRAY (IF INPUTS IS NUMPY ARRAY)
    if isinstance(X, np.ndarray):
        if y is None or class_names is None:
            print("please provide 'y' and 'class_names'")
            return
        n_samples = len(X)
        image_ids = range(n_samples)
        X = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)
    else:
        n_samples = len(filenames)
        image_ids = filenames
    if class_names is None:
        class_names = X.class_names
    
    element_spec = X.element_spec # element spec of original dataset (batched dataset)
    label_spec = element_spec[1] # label spec
    if len(label_spec.shape) == 2: # labels in categorical or binary mode
        if label_spec.shape[1] == 1: # binary mode
            ds = X.unbatch().map(lambda img, lbl: (img, int(lbl[0])))                   
        else: # categorical mode
            ds = X.unbatch().map(
                lambda img, lbl: (img, tf.math.argmax(lbl, output_type = 'int32'))
            )     
        # NOTE: after unbatching, the labels in 2D form is changed to 1D form 
    else:# labels in int mode
        ds = X.unbatch()
    # GET TRUE LABELS FROM THE DATASET
    true_labels = np.array(
        list(
            ds.map(lambda img, lbl: lbl).as_numpy_iterator()
        )
    )
    # right prediction -> True, wrong prediction -> False
    correct = true_labels == pred_labels 
    # true class names of image i
    true_class_names = [class_names[true_labels[i]] for i in range(len(true_labels))] 
    # predicted class names of image i
    pred_class_names = [class_names[pred_labels[i]] for i in range(len(pred_labels))] 
    
    # PRODUCE A pandas DataFrame
    all_pred_result = pd.DataFrame({
        'image_id': image_ids,
        'true_label': true_labels,
        'true_class_name': true_class_names,
        'pred_label': pred_labels,
        'pred_class_name': pred_class_names,
        'pred_proba': prob_labels,
        'correct': correct
    })
    # PLOT TOP WRONG PREDICTIONS
    # filter wrong result, and sort it (decreasing)
    wrong_pred_result = all_pred_result[all_pred_result['correct'] == False].sort_values(by = 'pred_proba', ascending = False)
    # original indices of images
    org_indices = np.array(wrong_pred_result.index)
    # indices of top wrong predictions
    top_wrong_indices = org_indices[:min(25, len(org_indices))]
    n = len(top_wrong_indices)
    # FIGURE CONFIGURATION
    r = math.ceil(math.sqrt(n)) # r image x r image of figure
    plt.figure(figsize = figsize)
    if cmap == 'grey': # cmap = 'grey' or plt.cm.binary for grey plotting
        cmap = plt.cm.binary
    j = 0 # image counter
    for i, (img, lbl) in enumerate(ds):
        if i in top_wrong_indices:
            j += 1
            #if j > 25: return
            plt.subplot(r,r,j)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.numpy().astype('uint8'), cmap = cmap)
            plt.title(f'label: {class_names[lbl]}')
            plt.xlabel( # predicted label
                f'prediction: {class_names[pred_labels[i]]}' +
                ' ({:2.2f}%)'.format(100*prob_labels[i])
            )
    # return all prediction or wrong prediction only
    if return_wrong_pred_only == True:
        return wrong_pred_result
    return all_pred_result

def f1_score(y_true, pred_probs, class_names=None, figsize=(8,15)):
    '''
    FUNCTIONALITIES: create a pandas DataFrame f1-score, and plot it
    ARGUMENTS:
    - y_true: true labels -> numpy array (1D, 2D) or tf dataset (test dataset, must include labels)
    - pred_probs: prediction probabilities -> 2D numpy array
    - class_names: name of classes -> list
    - figsize: figure size -> tuple of int
    RETURN: None
    USAGE:
    f1 = f1_score(y_true = y_test, pred_probs = test_pred_probs, class_names = ['cat', 'dog'], figsize = (4,6))
    f1 = f1_score(y_true = test_ds, pred_probs = test_pred_probs, class_names = test_ds.class_names, figsize = (6,15))
    '''
    # PRODUCE true_labels, class_names FROM y_true, class_names
    if isinstance(y_true, np.ndarray):# numpy array
        true_labels = y_true
    else:# tf dataset
        class_names = y_true.class_names
        true_labels = np.array(
            list(
                y_true.unbatch().map(lambda img, lbl: lbl).as_numpy_iterator()
            )
        )
    # if true_labels is 2D -> convert to 1D, else do nothing
    if true_labels.ndim == 2:
        if true_labels.shape[1] == 1: # (None, 1) -> binary classification
            if class_names is None:
                class_names = [0,1]
            true_labels = true_labels.flatten()
        else: # one hot encoding -> use argmax to find true labels in 1D array
            if class_names is None:
                class_names = list(
                    range(
                        len(true_labels.shape[1])
                    )
                )
            true_labels = np.argmax(true_lables, axis = 1)
    else:
        if class_names is None:
            class_names = list(
                range(len(np.unique(true_labels)))
            )

    # PRODUCE pred_labels, prob_labels FROM pred_probs
    if pred_probs.shape[1] == 1: # (None, 1) -> binary classification
        pred_labels_flattenned = pred_probs.flatten()
        pred_labels = np.where(pred_labels_flattenned > 0.5, 1, 0)
    else:
        pred_labels = np.argmax(pred_probs, axis = 1)

    # create classification report on testing data    
    cls_rep = classification_report(
        true_labels,           
        pred_labels,
        target_names=class_names,
        output_dict = True
    )
    f1_score_dict = {}
    # get f1-score only
    for cl_i, score_i in cls_rep.items():
        if cl_i == 'accuracy': break
        f1_score_dict[cl_i] = score_i['f1-score']
    f1_score = pd.DataFrame({
        'class_names': list(f1_score_dict.keys()),
        'f1-score': list(f1_score_dict.values())
    })
    # sort DataFrame by 'f1-score'
    f1_score.sort_values(by='f1-score', ascending = False, inplace = True)
    # plotting
    fig, ax = plt.subplots(figsize = figsize)
    score = ax.barh(y = range(len(f1_score)), width = f1_score['f1-score']) # horizontal bar
    ax.set(
        yticks = range(len(f1_score)),
        yticklabels = list(f1_score['class_names']),
        xticks = [0,0.2,0.4,0.6,0.8,1.0,1.2],
        title = 'f1-scores'
    )
    ax.invert_yaxis() # invert to sort decreasing
    # labels the bars
    for rect in score:
        width = rect.get_width()
        ax.text(
            1.05*width,# position
            rect.get_y() + rect.get_height()/1.5,# position
            f"{width:.2f}",#string
            ha='center', va='bottom' # horizontalalignment, verticalalignment
        )

        
def plot_confusion_matrix(y_true,
                          pred_probs,
                          class_names = None,
                          norm=False,
                          figsize=(15,15),
                          text_size = 10):
    '''
    FUNCTIONALITIES: create confusion matrix and plot it
    ARGUMENTS:
    - y_true: true labels -> numpy array (1D, 2D) or tf dataset (test dataset, must include labels)
    - pred_probs: prediction probabilities -> 2D numpy array
    - class_names: name of classes -> list
    - norm: normalize (True) or not (False) -> boolean
    - figsize: figure size -> tuple of int
    - text_size: size of text when displaying
    '''
    if isinstance(y_true, np.ndarray):
        true_labels = y_true
    else:
        class_names = y_true.class_names
        true_labels = np.array(
            list(
                y_true.unbatch().map(lambda img, lbl: lbl).as_numpy_iterator()
            )
        )
    if true_labels.ndim == 2:
        if true_labels.shape[1] == 1: # (None, 1) -> binary classification
            if class_names is None:
                class_names = [0,1]
            true_labels = true_labels.flatten()
        else: # one hot encoding -> use argmax to find true labels in 1D array
            if class_names is None:
                class_names = list(
                    range(
                        len(true_labels.shape[1])
                    )
                )
            true_labels = np.argmax(true_lables, axis = 1)
    else:
        if class_names is None:
            class_names = list(
                range(len(np.unique(true_labels)))
            )
    if pred_probs.shape[1] == 1: # (None, 1) -> binary classification
        pred_labels_flattenned = pred_probs.flatten()
        pred_labels = np.where(pred_labels_flattenned > 0.5, 1, 0)
    else:
        pred_labels = np.argmax(pred_probs, axis = 1)
    # confusion matrix (x axis : predicted labels, y axis: true labels)
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype("float")/cm.sum(axis=1)[:, np.newaxis] # normalize it (by row)
    fig, ax = plt.subplots(figsize = figsize)
    cax = ax.matshow(cm, cmap = plt.cm.Blues)
    fig.colorbar(cax)
    ax.set(
        title = 'confusion matrix',
        xlabel = 'predicted labels',
        ylabel = 'true labels',
        xticks = range(len(class_names)),
        xticklabels = class_names,
        yticks = range(len(class_names)),
        yticklabels = class_names
    )
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)
    # thresold for white and black text
    threshold = (cm.max() + cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.2f}%)",
                  horizontalalignment="center",
                  color="white" if cm[i, j] > threshold else "black",
                  size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                  horizontalalignment="center",
                  color="white" if cm[i, j] > threshold else "black",
                  size=text_size)
# END OF IMAGE CLASSIFICATION UTILITIES
####################################################
