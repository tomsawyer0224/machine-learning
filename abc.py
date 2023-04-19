import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import string
import re

def f1_score(y_true,pred_probs,class_names=None,figsize=(10,25)):
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
    cls_rep = classification_report(
        true_labels,           
        pred_labels,
        target_names=class_names,
        output_dict = True
    )
    f1_score_dict = {}
    for cl_i, score_i in cls_rep.items():
        if cl_i == 'accuracy': break
        f1_score_dict[cl_i] = score_i['f1-score']
    f1_score = pd.DataFrame({
        'class_names': list(f1_score_dict.keys()),
        'f1-score': list(f1_score_dict.values())
    })
    f1_score.sort_values(by='f1-score', ascending = False, inplace = True)
    
    fig, ax = plt.subplots(figsize = figsize)
    score = ax.barh(y = range(len(f1_score)), width = f1_score['f1-score'])
    ax.set(
        yticks = range(len(f1_score)),
        yticklabels = list(f1_score['class_names']),
        xticks = [0,0.2,0.4,0.6,0.8,1.0,1.2],
        title = 'f1-scores'
    )
    ax.invert_yaxis()
    for rect in score:
        width = rect.get_width()
        ax.text(
            1.05*width,# position
            rect.get_y() + rect.get_height()/1.5,#position
            f"{width:.2f}",#string
            ha='center', va='bottom' #horizontalalignment, verticalalignment
        )
        
    def plot_confusion_matrix(y_true,
                          pred_probs,
                          class_names = None,
                          norm=False,
                          figsize=(15,15),
                          text_size = 15):
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
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype("float")/cm.sum(axis=1)[:, np.newaxis] # normalize it
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
