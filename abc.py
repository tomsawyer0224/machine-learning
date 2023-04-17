import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import string
import re

def predict_and_plot(image, image_size, model, class_names):
    '''
    FUNCTIONALITIES: 
    - predict an image
    - plot result
    ARGUMENTS:
    - image: an image -> numpy array, tf tensor of shape (height, width, channels), file name, url
    - image_size: image will be resized to image_size (without channels)
    - model: keras classification model
    - class_names: name of classes -> list
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
        pred_label = 1 if (pred[0][0] > 0.5) else 0
    else:# categorical classification
        pred_label = np.argmax(pred, axis = 1)[0]# get label
    plt.imshow(img[0].numpy().astype('uint8'))# show image
    plt.xlabel(f'predict: {class_names[pred_label]}')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    
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
    
    # PRODUCE images, true_labels, class_names FROM X, y, class_names
    # we will create a new variable image_id (index of image if X is numpy array or file name of image if X is dataset)
    if isinstance(X, np.ndarray):
        images = X
        true_labels = y
        image_id = range(len(images))
    else:
        if class_names is None:
            class_names = X.class_names
        images = []
        true_labels = []
        for img, lbl in X.unbatch():
            images.append(img)
            true_labels.append(lbl)
        images = np.asarray(images)
        true_labels = np.asarray(true_labels)
        image_id = filenames
    #-----------------------------------------------------------------
    # we must provid true_labels
    if true_labels is None:
        print('please provide labels!')
        return
    
    # if true_labels is 2D -> use argmax to transform it into 1D array
    # if we don't provide class_names -> use class id instead
    if true_labels.ndim == 2:
        if class_names is None:
            class_names = list(range(true_labels.shape[1]))
        true_labels = np.argmax(true_labels, axis = 1)
    else:
        if class_names is None:
            class_names = list(
                range(len(np.unique(true_labels)))
            )
    
    correct = true_labels == pred_labels # right prediction -> True, wrong prediction -> False
    true_class_names = [class_names[true_labels[i]] for i in range(len(true_labels))] # true class names of image i
    pred_class_names = [class_names[pred_labels[i]] for i in range(len(pred_labels))] # predicted class names of image i
    # PRODUCE A pandas DataFrame
    all_pred_result = pd.DataFrame({
        'image_id': image_id,
        'true_label': true_labels,
        'true_class_name': true_class_names,
        'pred_label': pred_labels,
        'pred_class_name': pred_class_names,
        'pred_proba': prob_labels,
        'correct': correct
    })
    # filter wrong result, and sort it (decreasing)
    wrong_pred_result = all_pred_result[all_pred_result['correct'] == False].sort_values(by = 'pred_proba', ascending = False)
    # original indices of images
    org_indices = np.array(wrong_pred_result.index)
    # indices of top wrong predictions
    top_wrong_indices = org_indices[:min(25, len(org_indices))]
    # plotting
    n = len(top_wrong_indices)
    r = math.ceil(math.sqrt(n)) # r image x r image of figure
    plt.figure(figsize = figsize)
    if cmap == 'grey': # cmap = 'grey' or plt.cm.binary for grey plotting
        cmap = plt.cm.binary
    for i in range(n):
        idx = top_wrong_indices[i]
        plt.subplot(r,r,i+1)
        plt.imshow(images[idx].astype('uint8'), cmap = cmap)
        plt.title(f'label: {class_names[true_labels[idx]]}') # true label
        plt.xticks([])
        plt.yticks([])
        plt.xlabel( # predicted label
            f'prediction: {class_names[pred_labels[idx]]}' +
            ' ({:2.2f}%)'.format(100*prob_labels[idx])
        )
    plt.show()
    # return all prediction or wrong prediction only
    if return_wrong_pred_only == True:
        return wrong_pred_result
    return all_pred_result
