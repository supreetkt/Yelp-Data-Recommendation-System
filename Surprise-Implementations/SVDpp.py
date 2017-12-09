import pandas as pd
from surprise import SVD, evaluate, print_perf
from surprise.prediction_algorithms.matrix_factorization import SVDpp
from surprise import Reader, Dataset, dataset
from surprise.evaluate import GridSearch

outputpath= r'C:\Users\Guest123\Desktop\741 - Data Mining\Project\sampled_submission.csv'
inputpath = r'C:\Users\Guest123\Desktop\Data Mining\train_rating.txt'

class MyDataset(dataset.DatasetAutoFolds):
    def __init__(self, df, reader):
        self.raw_ratings = [(uid, iid, r, None) for (uid, iid, r) in
                            zip(df['user_id'], df['business_id'], df['rating'])]
        self.reader=reader

df = pd.read_csv(inputpath)

#drop the training_id
df = df.iloc[:,1:]

# Defining the format
reader = Reader(line_format='user item rating timestamp', sep=',')

# Loading data from the file using the reader format
data = MyDataset(df, reader)

# Split data into 5 folds
data.split(n_folds=5)

param_grid = {'n_epochs': [5, 10, 15, 20, 25, 30, 50, 100], 'lr_all': [0.001, 0.002, 0.005, 0.1, 0.2, 0.3, 0.5],
              'reg_all': [0, 0.1, 0.5, 0.02, 0.4, 0.6, 0.8, 1]}

#grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'MAE'], verbose=1)

#grid_search.fit(dataset.data, dataset.target)
#print(rsearch)

grid_search = GridSearch(SVDpp, param_grid, measures=['RMSE', 'MAE'], verbose=True)
grid_search.evaluate(data)