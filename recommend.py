import pickle
import json
import pandas as pd
from lightfm.cross_validation import random_train_test_split


#import the dataset
df_csv = pd.read_csv("./data/data_sample.csv")

# Drop out books with less than 6 interactions

book_filter = (df_csv["book_id"].value_counts()>5).to_dict()
book_filter = df_csv["book_id"].map(book_filter)
df_csv = df_csv.loc[book_filter,:]
print("sparsity is ",
                    100-(df_csv.shape[0]/(df_csv['book_id'].max()*df_csv['user_id'].max())*100))
x_train,x_test = random_train_test_split(user_mat)