#Running version
import pandas as pd
from surprise import SVD, evaluate
from surprise import Reader, dataset

outputpath= r'C:\Users\Guest123\Desktop\741 - Data Mining\Project\sampled_submission.csv'
inputpath_train = r'C:\Users\Guest123\Desktop\741 - Data Mining\Project\train_rating.txt'
inputpath_test = r'C:\Users\Guest123\Desktop\741 - Data Mining\Project\test_rating.txt'

class MyDataset(dataset.DatasetAutoFolds):
    def __init__(self, df, reader):
        self.raw_ratings = [(uid, iid, r, None) for (uid, iid, r) in
                            zip(df['user_id'], df['business_id'], df['rating'])]
        self.reader=reader

df = pd.read_csv(inputpath_train)

#drop the training_id
df = df.iloc[:,1:]

# Defining the format
reader = Reader(line_format='user item rating timestamp', sep=',')

# Loading data from the file using the reader format
data = MyDataset(df, reader)

# Split data into 5 folds
data.split(n_folds=5)

algo = SVD(n_epochs= 5)#, lr_all= 0.1, reg_all= 0.1)
evaluate(algo, data, measures=['RMSE', 'MAE'])

# Retrieve the trainset.
trainset = data.build_full_trainset()

#train!
algo.train(trainset)

#preparing test set
df = pd.read_csv(inputpath_test)#train_rating.txt')
df.insert(2, 'rating', 0, allow_duplicates=True)
reader = Reader(line_format='user item rating timestamp', sep=',')#test_id,user_id,business_id,date
testset = MyDataset(df, reader)

#test_results
test_results = algo.test(testset.construct_testset(raw_testset=testset.raw_ratings), verbose=False)
test_results = pd.DataFrame.from_records(test_results)
test_results2 = test_results.rename(columns={0: 'user_id', 1: 'item_id', 2: 'default_rating',3: 'rating', 4:'details'})
#drop default_rating

ratings = test_results2['rating']
testIDs_df = df['test_id']
final_results = pd.concat([testIDs_df, ratings], axis=1)
final_results.to_csv(outputpath, index=False)