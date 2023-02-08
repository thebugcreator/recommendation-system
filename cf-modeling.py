import pandas as pd
from lightfm import LightFM
from scipy import sparse
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score
from pandas.api.types import CategoricalDtype
import pickle

#import the dataset
df_csv = pd.read_csv("./data/data_sample.csv")

# Drop out books with less than 6 interactions

book_filter = (df_csv["book_id"].value_counts()>5).to_dict()
book_filter = df_csv["book_id"].map(book_filter)
df_csv = df_csv.loc[book_filter,:]

print("sparsity is ",
                    100-(df_csv.shape[0]/(df_csv['book_id'].max()*df_csv['user_id'].max())*100))

# build the scipy csr matrix
users = df_csv["user_id"].unique()
movies = df_csv["book_id"].unique()
shape = (len(users), len(movies))
user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
movie_cat = CategoricalDtype(categories=sorted(movies), ordered=True)
user_index = df_csv["user_id"].astype(user_cat).cat.codes
movie_index = df_csv["book_id"].astype(movie_cat).cat.codes
coo = sparse.coo_matrix((df_csv["rating"], (user_index, movie_index)), shape=shape)
user_mat = coo.tocsr()

print(f"Number of book used: {shape[0]}")
print(f"Number of users: {shape[1]}")

# Split data in train/test

x_train,x_test = random_train_test_split(user_mat)

# Modelling LightFM

model = LightFM(loss='warp',
                random_state=2016,
                learning_rate=0.7,
                no_components=300,
                learning_schedule="adadelta",
                user_alpha=0.0005)

model = model.fit(x_train,
                  epochs=10,
                  num_threads=16, verbose=False)

with open('model.pickle', 'wb') as fle:
    pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)

auc = auc_score(model, x_test, num_threads=4).mean()
print(f"AUC is :  {auc}")