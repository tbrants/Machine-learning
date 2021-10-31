from functions import *
from implementations import *

dirname = os.path.dirname('__file__')
DATA_TRAIN_PATH = os.path.join(dirname, 'data/train.csv')
y, tx, ids = load_csv_data(DATA_TRAIN_PATH)



