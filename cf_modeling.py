import pandas as pd
from scipy import sparse
from pandas.api.types import CategoricalDtype
import pickle
import json
import numpy as np

def data_to_csr(df_csv):

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

    return user_mat, movie_index


def get_model(model, x_train =None, train=False):

    if train:

        model = model.fit(x_train,
                        epochs=10,
                        num_threads=16, verbose=False)
        print("Model trained and save at ./lightfm-model/")
        with open('./lightfm-model/model.pickle', 'wb') as fle:
            pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        model = pickle.loads("./lightfm-model/model.pickle")
    return model


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
