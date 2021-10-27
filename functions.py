
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from proj1_helpers import *



" FEATURE MANIPULATION METHODS"

def build_poly(x, degree):
    #prof flammarion said it is possible, but i would wait a bit with doing it
    return x


def manipulate_missing_values(tx):
    tx[tx == -999.0] = np.nan
    avg_column = np.array([np.nanmean(tx, axis=0)])
    avg_column_ = np.repeat(avg_column, tx.shape[0], axis=0)
    one_dim_indices_to_subst = np.where(np.isnan(tx))
    two_dim_indices_to_subst = np.squeeze(np.dstack((one_dim_indices_to_subst[0],one_dim_indices_to_subst[1])), axis=0)
    tx[two_dim_indices_to_subst] = avg_column_[two_dim_indices_to_subst]
    return tx


# normalize features
# this could be standarize...
def normalize_features(tx):
    for j in range(tx.shape[1]):
        col = tx[:,j]
        tx[:,j] = (col-col.min())/(col.max()-col.min())
    return tx


# balance data because there were lots of '-1' labels
def balance_data(tx, y, perc_plus_labels):
    plus_indices = np.array(np.where(y==1)[0])
    min_indices = np.array(np.where(y==-1)[0])
    print('Before balancing')
    print('%d (%.2f)%% 1 labels in training data'%(len(plus_indices), len(plus_indices)/(len(plus_indices)+len(min_indices))*100))
    print('%d (%.2f)%% -1 labels in training data'%(len(min_indices), len(min_indices)/(len(plus_indices)+len(min_indices))*100))

    #make balanced train datasets
    min_labels_size = int(len(plus_indices)*(1-perc_plus_labels)/perc_plus_labels) #calculate number of majority class (-1) labels
    random_indices = np.random.choice(min_indices, min_labels_size, replace=False) #choose random number of indices
    a = tx[random_indices.astype(int)]
    tx = np.append(a,tx[plus_indices.astype(int)])
    b = y[random_indices.astype(int)]
    y = np.append(b, y[plus_indices.astype(int)])
    print('After balancing')
    plus_indices = np.where(y==1)[0]
    min_indices = np.where(y==-1)[0]
    print('%d (%.2f)%% 1 labels in training data'%(len(plus_indices), len(plus_indices)/(len(plus_indices)+len(min_indices))*100))
    print('%d (%.2f)%% -1 labels in training data'%(len(min_indices), len(min_indices)/(len(plus_indices)+len(min_indices))*100))
    return tx.reshape(-1,30), y

" MACHINE LEARNING METHODS"
def compute_error(y, tx, w):
    e = y-np.matmul(tx, w)
    return e


def compute_loss(y, tx, w):
    N = len(y)
    e = compute_error(y, tx, w)
    mse = 1/(2*N)*np.sum(e**2)
    rmse = np.sqrt(2*mse)
    return rmse


def compute_gradient(y, tx, w):
    N = len(y)
    e = compute_error(y, tx, w)
    grad = (-1/N)*np.matmul(tx.T,e)
    return grad


#gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w -= gamma*grad
    loss = compute_loss(y, tx, w)
    print("Gradient Descent: RMSE is ", loss)
    return w, loss


def compute_stoch_gradient(y, tx, w):
    N = len(y)
    e = compute_error(y, tx, w)
    grad = (-1/N)*np.matmul(tx.T,e)
    return grad


#stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma,  batch_size = 1):
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
    N = len(y)
    w = np.linalg.solve(np.matmul(tx.T, tx), np.matmul(tx.T,y))
    loss = compute_loss(y, tx, w)
    print("Least squares: RMSE is ", loss)
    return w, loss


#ridge regression
def ridge_regression(y, phi, lambda_):
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
    eps = 1e-6
    pred = sigmoid(np.matmul(tx, w))
    a = np.matmul(y.T, np.log(pred+eps)) 
    b = np.matmul((1-y).T, np.log(1-pred+eps))
    loss = a+b
    return np.sum(- loss)


def calculate_gradient_lr(y, tx, w):
    pred = sigmoid(np.matmul(tx, w)) 
    grad = np.matmul(tx.T,(pred - y))
    return grad


def penalized_logistic_regression(y, tx, w, lambda_):
    loss = calculate_loss_lr(y, tx, w) + lambda_*np.sum(w.T.dot(w))
    gradient = calculate_gradient_lr(y,tx,w)+2*lambda_*w
    return loss, gradient


#logistic regression gradient descent
def logistic_regression_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for iter in range(max_iters):
        grad = calculate_gradient_lr(y, tx, w)
        w -= gamma*grad
        loss = calculate_loss_lr(y, tx, w)
    print("GD Logistic Regression: RMSE is ", loss)
    return w, loss


#regularized logistic regression
def logistic_regression_reg(y, tx, initial_w, max_iters, gamma, lambda_):
    w = initial_w
    for iter in range(max_iters):
        grad = penalized_logistic_regression(y, tx, w, lambda_)[1]
        w -= gamma*grad
    loss = penalized_logistic_regression(y, tx, w, lambda_)[0]
    print("Reg Logistic Regression: RMSE is ", loss)
    return w, loss



" CROSS VALIDATION METHODS "

def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def get_train_test_sets(y, x, k, degree, seed):
    k_indices=build_k_indices(y, k, seed)
    
    x_train_0, y_train_0 = x[k_indices[:k-1].ravel()], y[k_indices[:k-1].ravel()]
    x_train_1, y_train_1 = x[k_indices[k:].ravel()], y[k_indices[k:].ravel()]
   
    if x_train_0.shape[0] == 0:
        x_train = x_train_1
        y_train = y_train_1
    if x_train_1.shape[0] == 0:
        x_train = x_train_0
        y_train = y_train_0
    else:
        x_train, y_train = np.concatenate((x_train_0, x_train_1), axis=0), np.concatenate((y_train_0, y_train_1), axis=0)
        
    x_test, y_test = x[k_indices[k-1]], y[k_indices[k-1]]

    #phi_train = build_poly(x_train, degree)
    #phi_test = build_poly(x_test, degree)
    phi_train = x_train
    phi_test = x_test
    
    return phi_train, y_train, phi_test, y_test


def call_cross_validation(y, x, k, degree, seed, opt_method, initial_w, max_iters, gamma, lambda_):
    # Returns the average of the results
    w_kfold = []
    loss_train_kfold = []
    loss_test_kfold = []
    f1_kfold = []
    acc_kfold = []
    
    for ki in range(k):
        seed = np.random.randint(100)
        
        #the seed makes each k_indeces different from the one before I believe...
        phi_train, y_train, phi_test, y_test = get_train_test_sets(y, x, k, degree, seed) 
        
        if opt_method == least_squares_GD:
            print('least_squares_GD')
            w, loss = opt_method(y_train, phi_train, initial_w, max_iters, gamma)
            
        elif opt_method == least_squares_SGD:
            print('least_squares_SGD')
            w, loss = opt_method(y_train, phi_train, initial_w, max_iters, gamma,  batch_size = 1)
            
        elif opt_method == least_squares:
            print('least_squares')
            w, loss = opt_method(y_train, phi_train)
         
        elif opt_method == ridge_regression:
            print('ridge_regression')
            w, loss = opt_method(y_train, phi_train, lambda_)
        
        elif opt_method == logistic_regression_gd:
            print('logistic_regression_gd')
            w, loss = opt_method(y_train, phi_train, initial_w, max_iters, gamma)
        
        elif opt_method == logistic_regression_reg:
            print('logistic_regression_reg')
            w, loss = opt_method(y_train, phi_train, initial_w, max_iters, gamma, lambda_)
            

        else:
            w, loss = (None , None)
            test_loss = None
            print("Method not found")
        
        test_loss = compute_loss(y_test, phi_test, w)
        y_pred = predict_labels(w, phi_test)
        f1_value = f1(y_test, y_pred)
        acc_value = accuracy(y_test, y_pred)
        
        f1_kfold.append(f1_value)
        acc_kfold.append(acc_value)
        w_kfold.append(w)
        loss_train_kfold.append(loss)
        loss_test_kfold.append(test_loss)
        
        
    return np.mean(w_kfold,axis=0), np.mean(loss_train_kfold), np.mean(loss_test_kfold), np.mean(f1_kfold), np.mean(acc_kfold) 




"PERFORMANCE EVALUATION METHODS"

# false positives and false negatives evaluation
def f1(y_true, y_pred):
    TP = len(np.where((y_true == 1) & (y_pred == 1))[0])
    FP = len(np.where((y_true == -1) & (y_pred == 1))[0])
    TN = len(np.where((y_true == -1) & (y_pred == -1))[0])
    FN = len(np.where((y_true == 1) & (y_pred == -1))[0])
    print('TP = ',TP,'; FP = ',FP,'; TN = ',TN,'; FN = ',FN )
    Recall = TP/(TP+FN)
    Precision = TP/(TP+FP)
    F = 2*Recall*Precision/(Recall + Precision)
    print('F1 =',F)
    return F


# accuracy, measures the correctly identified casaes, should be near 1
def accuracy(y_true, y_pred):
    TP = len(np.where((y_true == 1) & (y_pred == 1))[0])
    FP = len(np.where((y_true == -1) & (y_pred == 1))[0])
    TN = len(np.where((y_true == -1) & (y_pred == -1))[0])
    FN = len(np.where((y_true == 1) & (y_pred == -1))[0])
    A = (TP+TN)/(TP+FP+TN+FN)
    print('Accuracy =',A)
    return A


def hyperparameter_tuning(y,tx,k,seed,meth,initw,max_iterations,degrees,gammas,lambdas):
    pd_filled = pd.DataFrame()

    for lam in lambdas:
        for ga in gammas:
            for deg in degrees:
                for ma in max_iterations:
                    w, rmse_tr, rmse_te, f1, acc = call_cross_validation(y, tx, k, 
                                                         deg, seed,
                                                         meth, 
                                                         initw, 
                                                         ma, 
                                                         ga, 
                                                         lam)
                    dictio = {'max_iters':ma,'lambda':lam,'gamma':ga,'degree':deg,'out_w':w, 'rmse_tr':rmse_tr, 'rmse_te':rmse_te, 'f1':f1, 'acc':acc}
                    pd_filled = pd_filled.append(dictio, ignore_index=True)
    return pd_filled
