import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from helpers import *
from functions import *

" FEATURE MANIPULATION METHODS"

    #polynomial feature augmentation
def build_poly(x, degree): 
    """
    x: training data
    degree: polynomial augmentation degree
    
    returns: log normalized tx
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def manipulate_missing_values(tx):
    """
    tx: training data
    
    returns: tx without useless arrays and with -999 replaced by mean of the row
    """
    tx[tx == -999.0] = np.nan
    tx = pd.DataFrame(tx)
    tx=tx.dropna(axis=1, how='all')
    tx = tx.to_numpy()
    avg_column = np.array([np.nanmean(tx, axis=0)])
    avg_column_ = np.repeat(avg_column, tx.shape[0], axis=0)
    one_dim_indices_to_subst = np.where(np.isnan(tx))
    two_dim_indices_to_subst = zip(one_dim_indices_to_subst[0],one_dim_indices_to_subst[1])
    for t in two_dim_indices_to_subst:
        tx[t]=avg_column_[t]
    
    return tx

#standardize features
def standardize_features(tx):
    """
    tx: training data
    
    returns: standardized tx
    """
    for j in range(tx.shape[1]):
        col = tx[:,j]
        tx[:,j] = (col-np.mean(col))/np.std(col)
    return tx

#lognormalize features
def lognormalize_features(tx):
    """
    tx: training data
    
    returns: log normalized tx
    """
    skewed_indices = [0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 29] ## indices obtained by looking at the data
    tx[:, skewed_indices] = np.log1p(tx[:, skewed_indices])
    return tx

#process data
def data_process(x_tr,x_te,processing_level):
    """
    x_tr: training data
    x_te: test data
    processing_level: amount of processing needed
    
    returns: processed x_tr and x_te
    """
    
    
    retur
    if processing_level == 'std': 
        #standardize
        x_tr = standardize_features(x_tr)
        x_te = standardize_features(x_te)
    
    if (processing_level == 'log') | (processing_level == 'std'):
        #lognormalize
        x_tr = lognormalize_features(x_tr)
        x_te = lognormalize_features(x_te)
        
    #average -999 values
    x_tr = manipulate_missing_values(x_tr)
    x_te = manipulate_missing_values(x_te)
    
    return x_tr, x_te
#Get the index of the jet number group (cfr. report)
def get_index_jet(tx):
    """
    tx: training data
    
    returns: 3 datasets grouped per jet number category
    """
    jeti_0 = np.where(tx[:, 22] == 0)[0]
    jeti_1 = np.where(tx[:, 22] == 1)[0]
    jeti_2 = np.where(tx[:, 22] >= 2)[0]
    return [jeti_0, jeti_1, jeti_2]

#create subsets of data according to the jetnumber
def create_subdata_jetnumber(tx, y , tx_test, processing_level):
    """
    tx: training data
    y: labels
    tx_test: test data
    processing_level: string which tells the amount of processing needed to be done
    
    returns: [3,n]-matrices with X_TRAIN, Y_TRAIN and X_TEST data per jet category
    """
    jeti_train = get_index_jet(tx)  ## Separation of the data using the jet number
    jeti_test = get_index_jet(tx_test)
    for i in range(3):
        if i == 0:
            xtr_0 = tx[jeti_train[i]]
            ytr_0 = y[jeti_train[i]]
            xte_0 = tx_test[jeti_test[i]]
            xtr_0, xte_0 = data_process(xtr_0, xte_0, processing_level)
        if i == 1:
            xtr_1 = tx[jeti_train[i]]
            ytr_1 = y[jeti_train[i]]
            xte_1 =  tx_test[jeti_test[i]]
            xtr_1, xte_1 = data_process(xtr_1, xte_1, processing_level) 
        if i == 2:
            xtr_2 = tx[jeti_train[i]]
            ytr_2 = y[jeti_train[i]]
            xte_2 = tx_test[jeti_test[i]]
            xtr_2, xte_2 = data_process(xtr_2, xte_2, processing_level)
    X_TRAIN_jets = [xtr_0, xtr_1, xtr_2]
    Y_TRAIN_jets = [ytr_0,ytr_1,ytr_2]
    X_TEST = [xte_0,xte_1,xte_2]
    return X_TRAIN_jets, Y_TRAIN_jets, X_TEST

" CROSS VALIDATION METHODS "

def build_k_indices(y, k_fold, seed):
    """
    y: labels
    k_fold: amount of cross_validation folds
    
    returns: [k,y.shape[0]]-matrix of random rows
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def get_train_test_sets(y, x, k, degree, seed):
    """
    y: labels
    x: feature data
    k: amount of cross_validation folds
    degree: polynomial augmentation degree
    
    returns: random selected x_train, y_train, x_test and y_test
    """
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

    phi_train = build_poly(x_train, degree)
    phi_test = build_poly(x_test, degree)
    
    return phi_train, y_train, phi_test, y_test


def call_cross_validation(y, x, k, degree, seed, opt_method, initial_w, max_iters, gamma, lambda_):
    """
    y: labels
    x: feature data
    k: amount of cross_validation folds
    degree: polynomial augmentation degree
    opt_meth: machine learning method to use
    initial_w: array of factors to multiply by initial weight array
    max_iters: maxiumum amount of iterations
    gamma: learning rate
    lambdas: regularization parameter
    
    returns: pandas dataframe, optimal weights, optimal degrees
    """
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
    """
    calculates f1 from the true and predicted labels
    returns accuracy
    """
    TP = len(np.where((y_true == 1) & (y_pred == 1))[0])
    FP = len(np.where((y_true == -1) & (y_pred == 1))[0])
    TN = len(np.where((y_true == -1) & (y_pred == -1))[0])
    FN = len(np.where((y_true == 1) & (y_pred == -1))[0])
    print('TP = ',TP,'; FP = ',FP,'; TN = ',TN,'; FN = ',FN )
    if(TP+FN == 0):
        Recall = 0
    else:
        Recall = TP/(TP+FN)
    if(TP+FP == 0):
        Precision = 0
    else:
        Precision = TP/(TP+FP)
    if(Recall + Precision == 0):
        F = 0
    else:
        F = 2*Recall*Precision/(Recall + Precision)
    print('F1 =',F)
    return F


# accuracy, measures the correctly identified cases, should be near 1
def accuracy(y_true, y_pred):
    """
    calculates accuracy from the true and predicted labels
    returns accuracy
    """
    TP = len(np.where((y_true == 1) & (y_pred == 1))[0])
    FP = len(np.where((y_true == -1) & (y_pred == 1))[0])
    TN = len(np.where((y_true == -1) & (y_pred == -1))[0])
    FN = len(np.where((y_true == 1) & (y_pred == -1))[0])
    if(TP+FP+TN+FN == 0):
        A = 0
    else:
        A = (TP+TN)/(TP+FP+TN+FN)
    print('Accuracy =',A)
    return A

def hyperparameter_tuning(y,tx,k,seed,meth,max_iterations,degrees,gammas,lambdas,initial):
    """
    tunes hyperparameters by using grid_search, uses accuracy as a measure
    
    y: labels
    tx: feature data
    k: amount of cross_validation folds
    meth: machine learning method to use
    degrees: array of polynomial augmentation degree
    gammas: array of learning rates
    lambdas: array of regularization parameters
    initial: array of factors to multiply by initial weight array
    
    returns: pandas dataframe, optimal weights, optimal degrees
    """
    pd_filled = pd.DataFrame()
    max_acc = 0
    opt_w = []
    opt_degree = 0
    for lam in lambdas:
        for ga in gammas:
            for deg in degrees:
                for ma in max_iterations:
                    for ini in initial:
                        initw = np.ones(tx.shape[1] * deg + 1)*ini
                        w, rmse_tr, rmse_te, f1, acc = call_cross_validation(y, tx, k, 
                                                             deg, seed,
                                                             meth, 
                                                             initw, 
                                                             ma, 
                                                             ga, 
                                                             lam)
                        if acc > max_acc:
                            maw_acc = acc
                            opt_w = w
                            opt_degree = deg
                        dictio = {'max_iters':ma,'lambda':lam,'gamma':ga,'degree':deg,'out_w':w, 'rmse_tr':rmse_tr, 'rmse_te':rmse_te, 'f1':f1, 'acc':acc, 'factors':ini}
                        pd_filled = pd_filled.append(dictio, ignore_index=True)
    return pd_filled, opt_w, opt_degree 
