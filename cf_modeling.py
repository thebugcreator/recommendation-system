import pandas as pd
from lightfm import LightFM
from scipy import sparse
from lightfm.evaluation import auc_score
from pandas.api.types import CategoricalDtype
import pickle
import json
import numpy as np

def data_csr(df_csv):

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

    return user_mat



# Split data in train/test

# Modelling LightFM

model = LightFM(loss='warp',
                random_state=2016,
                learning_rate=0.7,
                no_components=300,
                learning_schedule="adadelta",
                user_alpha=0.0005)

def get_model(model, x_train =None, train=False):

    if train:

        model = model.fit(x_train,
                        epochs=10,
                        num_threads=16, verbose=False)
        print("Model trained and save at ./lightfm-model/")
        with open('model.pickle', 'wb') as fle:
            pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        model = pickle.loads("./lightfm-model/model.pickle")
    return model



auc = auc_score(model, x_test, num_threads=4).mean()
print(f"AUC is :  {auc}")

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

def sample_recommendation(model, user_ids, random_user=None):

    n_users, n_items = user_mat.shape
    if random_user is not None:
        user_ids = user_ids[:1]
    #print(user_ids)
    for user_id in user_ids:
        user_movie = [item_labels[movie_dict_r[k]] for k in user_mat[user_id].indices]
        known_positives = [item_labels[movie_dict_r[k]] for k in user_mat[user_id].indices]

        scores = model.predict(user_id, np.arange(n_items), item_features=random_user)
        top_items = [item_labels[movie_dict_r[k]] for k in np.argsort(-scores)]

        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:5]:
            id = book_map.iloc[item_labels[movie_dict_r[x]],1]
            x = labels[str(id)]
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:5]:
            id = book_map.iloc[item_labels[movie_dict_r[x]],1]
            x = labels[str(id)]
            print("        %s" % x)

new_user = np.random.randint(5, size=(user_mat.shape[1], 1))
sample_recommendation(model, user_ids=[222,23], random_user=None)
