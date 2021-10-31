from functions import *
from implementations import *

#set paths

dirname = os.path.dirname('__file__')
DATA_TRAIN_PATH = os.path.join(dirname, 'data/train.csv')
y, tx, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = os.path.join(dirname, 'data/test.csv') # TODO: download train data and supply path here 
_, tx_test, ids_test = load_csv_data(DATA_TEST_PATH)

#hyperparametertune, feature processing and writeing out results
def feature_proc_performance(tx, y, tx_test, processing_method):
    """
    tx: training data
    y: labels
    tx_test: test data
    processing method: amount of processing needed
    
    returns: none, prints results to csv files
    """
    X_TRAIN_jets, Y_TRAIN_jets, X_TEST = create_subdata_jetnumber(tx, y, tx_test, processing_method)

    #jet 0
    lambdas = [1e-6, 1e-7, 1e-8, 1e-9]
    gammas = [0]
    degrees = [1,2,3,4,5,6,7,8,9,10]
    max_iterations =  [0]
    factors = [0]
    
    pd_filled_RR_tr1, w_0, d_0 = hyperparameter_tuning(Y_TRAIN_jets[0],X_TRAIN_jets[0],4,2,ridge_regression,max_iterations,degrees,gammas,lambdas,factors)
    
    #jet 1
    lambdas = [1e-6, 1e-7, 1e-8, 1e-9]
    gammas = [0]
    degrees = [1,2,3,4,5,6,7,8,9,10]
    max_iterations =  [0]
    factors = [0]
    
    pd_filled_RR_tr2, w_1, d_1 = hyperparameter_tuning(Y_TRAIN_jets[1],X_TRAIN_jets[1],4,2,ridge_regression,max_iterations,degrees,gammas,lambdas,factors)
    
    #jet 2
    lambdas = [1e-6, 1e-7, 1e-8, 1e-9]
    gammas = [0]
    degrees = [1,2,3,4,5,6,7,8,9,10]
    max_iterations =  [0]
    factors = [0]
    pd_filled_RR_tr3, w_2, d_2  = hyperparameter_tuning(Y_TRAIN_jets[2],X_TRAIN_jets[2],4,2,ridge_regression,max_iterations,degrees,gammas,lambdas,factors)
    
    #write to files
    x_test_augm0 = build_poly(X_TEST[0],d_0)
    y_pred_0 = predict_labels(w_0, x_test_augm0)

    x_test_augm1 = build_poly(X_TEST[1],d_1)
    y_pred_1 = predict_labels(w_1, x_test_augm1)

    x_test_augm2 = build_poly(X_TEST[2],d_2)
    y_pred_2 = predict_labels(w_2, x_test_augm2)

    jeti_0, jeti_1, jeti_2 = get_index_jet(tx_test)
    i0 = ids_test[jeti_0]
    i1 = ids_test[jeti_1]
    i2 = ids_test[jeti_2]
    indexes = np.concatenate((i0, i1, i2))
    
    y_labels_pred = np.concatenate((y_pred_0,y_pred_1, y_pred_2), axis=0)

    OUTPUT_PATH = os.path.join(dirname, 'data/pred_'+processing_method+'.csv') 
    create_csv_submission(indexes, y_labels_pred, OUTPUT_PATH)
    return 

#data splitted on jet nums, filtered out -999
feature_proc_performance(tx, y, tx_test, 'split')

#data splitted, -999 filtered and log normalized
feature_proc_performance(tx, y, tx_test, 'log')

#data splitted, -999 filtered, log normalized and standardized
feature_proc_performance(tx, y, tx_test, 'std')