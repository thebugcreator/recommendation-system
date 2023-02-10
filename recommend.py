from cf_modeling import data_to_csr, sample_recommendation, get_model
import pandas as pd
from lightfm.cross_validation import random_train_test_split
import json
from lightfm import LightFM
import numpy as np

#import the dataset
df_csv = pd.read_csv("./datasets/data_sample.csv")

# Drop out books with less than 6 interactions

book_filter = (df_csv["book_id"].value_counts()>5).to_dict()
book_filter = df_csv["book_id"].map(book_filter)
df_csv = df_csv.loc[book_filter,:]
print("sparsity is ",
                    100-(df_csv.shape[0]/(df_csv['book_id'].max()*df_csv['user_id'].max())*100))

user_mat, movie_index = data_to_csr(df_csv)

x_train,x_test = random_train_test_split(user_mat)

# get labels of books
item_labels = df_csv["book_id"]
# get books name
with open("items_labels.json","r") as f:
    labels = json.load(f)

# get mapping of books and labels
book_map = pd.read_csv("data/book_id_map.csv", sep=",")

# transform movie index into dict and reverse
movie_dict = movie_index.to_dict()
movie_dict_r = {v:k for k,v in movie_dict.items()}
model = LightFM(loss='warp',
                random_state=2016,
                learning_rate=0.7,
                no_components=300,
                learning_schedule="adadelta",
                user_alpha=0.0005)
model = get_model(model, x_train =None, train=False)
new_user = np.random.randint(5, size=(user_mat.shape[1], 1))
sample_recommendation(model, user_ids=[222,23], random_user=None)