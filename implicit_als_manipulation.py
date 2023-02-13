import pandas as pd
import numpy as np
import scipy.sparse as sparse
import random
import implicit
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Colaborative Filtering Model with Implicit")
    parser.add_argument("--model_folder", type=str, default="implicit_als_model/", help="Path to the model folder.")
    parser.add_argument("--type", default="user", const="all", nargs="?", help="Choose the subject for recommendation", choices=["user", "book"])
    parser.add_argument("query", type=int, help="ID of the subject.")
    args = parser.parse_args()
    
    model_folder = args.model_folder
    type = args.type
    query = args.query

	
    model_path = model_folder + "model.npz"
	model_data = np.load(als_model_file, allow_pickle=True)
	
    model = implicit.als.AlternatingLeastSquares(factors=200)
	model.item_factors = model_data['model.item_factors']
	model.user_factors = model_data['model.user_factors']
	model._YtY = model.item_factors.T.dot(model.item_factors)
	
    sparse_book_user = sparse.load_npz(model_folder + "book_user.npz")
    sparse_user_book = sparse.load_npz(model_folder + "user_book.npz")
    
    if type == "user":
        recommendation = model.recommend(query, sparse_user_book[query])
    elif type == "book":
        recommendation = model.recommend(query, sparse_book_user[query])
    else:
        print("Invalid type")
        exit()
    
    print(recommendation)