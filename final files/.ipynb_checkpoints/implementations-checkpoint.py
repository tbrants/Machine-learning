import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from helpers import *
from functions import *

" MACHINE LEARNING METHODS"
def compute_error(y, tx, w):
    """
    y: labels
    tx: training data
    w: weights
    
    returns: error
    """
    e = y-np.matmul(tx, w)
    return e


def compute_loss(y, tx, w):
    """
    y: labels
    tx: training data
    w: weights
    
    returns: loss
    """
    N = len(y)
    e = compute_error(y, tx, w)
    mse = 1/(2*N)*np.sum(e**2)
    rmse = np.sqrt(2*mse)
    return rmse


def compute_gradient(y, tx, w):
    """
    y: labels
    tx: training data
    w: weights
    
    returns: gradient
    """
    N = len(y)
    e = compute_error(y, tx, w)
    grad = (-1/N)*np.matmul(tx.T,e)
    return grad


#gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    y: labels
    tx: training data
    intitial_w: weights
    max_iters: maximum amount of iterations
    gamma: learning rate
    
    returns: weights, loss
    """
    w = initial_w
    print(max_iters)
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w -= gamma*grad
    loss = compute_loss(y, tx, w)
    print("Gradient Descent: RMSE is ", loss)
    return w, loss


def compute_stoch_gradient(y, tx, w):
    """
    y: labels
    tx: training data
    w: weights
    
    returns: gradient
    """
    N = len(y)
    e = compute_error(y, tx, w)
    grad = (-1/N)*np.matmul(tx.T,e)
    return grad


#stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma,  batch_size = 1):
    """
    y: labels
    tx: training data
    w: weights
    
    returns: gradient
    """
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for ymini, txmini in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
            grad = compute_gradient(ymini, txmini, w)
            w -= gamma*grad
    loss = compute_loss(ymini, txmini, w)
    print("Stochastic Gradient Descent: RMSE is ", loss)
    return w, loss


#least squares
def least_squares(y, tx):
    """
    y: labels
    tx: training data
    
    returns: weight, loss
    """
    N = len(y)
    w = np.linalg.solve(np.matmul(tx.T, tx), np.matmul(tx.T,y))
    loss = compute_loss(y, tx, w)
    print("Least squares: RMSE is ", loss)
    return w, loss


#ridge regression
def ridge_regression(y, phi, lambda_):
    """
    y: labels
    phi: training data
    w: weights
    
    returns: weight, loss
    """
    N=len(phi)
    lambda_acc=2*N*lambda_
    kwad = np.matmul(phi.T,phi)
    w = np.linalg.solve(kwad+lambda_acc*np.eye(kwad.shape[0]),np.matmul(phi.T,y))
    loss = compute_loss(y, phi, w)
    print("Ridge Regression: RMSE is ", loss)
    return w, loss


def sigmoid(t):
    return 1/(1 + np.exp(-t))


def calculate_loss_lr(y, tx, w):
    """
    y: labels
    tx: training data
    w: weights
    
    returns: loss
    """
    #change -1 values to 0 in order for loss to work
    y_ = y
    y_[y_]
    
    eps = 1e-6
    pred = sigmoid(np.matmul(tx, w))
    a = np.matmul(y.T, np.log(pred+eps)) 
    b = np.matmul((1-y).T, np.log(1-pred+eps))
    loss = a+b
    return np.sum(- loss)


def calculate_gradient_lr(y, tx, w):
    """
    y: labels
    tx: training data
    w: weights
    
    returns: gradient
    """
    pred = sigmoid(np.matmul(tx, w)) 
    grad = np.matmul(tx.T,(pred - y))
    return grad


def penalized_logistic_regression(y, tx, w, lambda_):
    """
    y: labels
    tx: training data
    w: weights
    lambda_: regularization parameter
    
    returns: gradient
    """
    loss = calculate_loss_lr(y, tx, w) + lambda_*np.sum(w.T.dot(w))
    gradient = calculate_gradient_lr(y,tx,w)+2*lambda_*w
    return loss, gradient


#logistic regression gradient descent
def logistic_regression_gd(y, tx, initial_w, max_iters, gamma):
    """
    y: labels
    tx: training data
    initial_w: weights
    max_iters: maximum amount of iterations
    gamma: learning rate
    
    returns: gradient
    """
    w = initial_w
    for iter in range(max_iters):
        grad = calculate_gradient_lr(y, tx, w)
        w -= gamma*grad
        loss = calculate_loss_lr(y, tx, w)
    print("GD Logistic Regression: RMSE is ", loss)
    return w, loss


#regularized logistic regression
def logistic_regression_reg(y, tx, initial_w, max_iters, gamma, lambda_):
    """
    y: labels
    tx: training data
    initial_w: weights
    max_iters: maximum amount of iterations
    gamma: learning rate
    lambda_: regularization parameter
    
    returns: gradient
    """
    w = initial_w
    for iter in range(max_iters):
        grad = penalized_logistic_regression(y, tx, w, lambda_)[1]
        w -= gamma*grad
    loss = penalized_logistic_regression(y, tx, w, lambda_)[0]
    print("Reg Logistic Regression: RMSE is ", loss)
    return w, loss