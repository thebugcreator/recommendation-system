import pandas as pd
import numpy as np
import scipy.sparse as sparse
import random
import implicit
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Colaborative Filtering Model with Implicit")
    parser.add_argument("--model_path", type=str, default="implicit_als_model/", help="Path to the model folder.")
    parser.add_argument("--type", default="user", const="all", nargs="?", help="Choose the subject for recommendation", choices=["user", "book"])
    parser.add_argument("query", type=int, help="ID of the subject.")
    args = parser.parse_args()
    
    model_path = args.model_path
    type = args.type
    query = args.query


    model_path = model_path + "model.npz"

    model = implicit.als.AlternatingLeastSquares(model_path)
    sparse_book_user = sparse.load_npz(model_path + "book_user.npz")
    sparse_user_book = sparse.load_npz(model_path + "user_book.npz")
    
    if type == "user":
        recommendation = model.recommend(query, sparse_user_book[query])
    elif type == "book":
        recommendation = model.recommend(query, sparse_book_user[query])
    else:
        print("Invalid type")
        exit()
    
    print(recommendation)