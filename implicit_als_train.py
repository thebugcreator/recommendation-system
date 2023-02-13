import pandas as pd
import numpy as np
import scipy.sparse as sparse
import random
import implicit 
import argparse

def calc_confidence(is_read, rating, is_reviewed, weights=(1,1,1)):
    x, y, z = weights
    bc = 1 # base confidence by the ALS model
    bc += is_read*x
    bc += rating*y
    bc += is_reviewed*z
    return bc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Colaborative Filtering Model with Implicit")
    parser.add_argument("--csv_path", type=str, default="datasets/goodreads_interactions.csv", help="Path to the csv interaction file.")
    parser.add_argument("--save_model", type=bool, default=True, help="Decide whether or not to save the model")
    parser.add_argument("--factors", type=int, default=20, help="Model factors")
    parser.add_argument("--regularization", type=float, default=0.1, help="Model regularisation")
    parser.add_argument("--iterations", type=int, default=20, help="Model iteration")
    parser.add_argument("--alpha", type=int, default=40, help="Model alpha")
    args = parser.parse_args()
    
    csv_path = args.csv_path
    save_model = args.save_model
    metric = args.metric
    factors = args.factors
    regularization = args.regularization
    iterations = args.iterations
    alpha = args.alpha
    
    print("Looking into the dataset at ", csv_path)
    # Load the interaction dataset
    df = pd.read_csv(csv_path)
    
    # Start creating the confidence matrix
    # df["confidence"] = [calc_confidence(*(row[["is_read","rating","is_reviewed"]].values)) for index, row in df.iterrows()]
    
    print("Number of records: ", df.shape[0])
    
    data = df[["user_id", "book_id", "rating"]]
    
    sparse_item_user = sparse.csr_matrix((data['rating'].astype(float), (data['book_id'], data['user_id'])))
    sparse_user_item = sparse.csr_matrix((data['rating'].astype(float), (data['user_id'], data['book_id'])))
    
    # Building the model
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)
    alpha_val = alpha
    data_conf = (sparse_item_user * alpha_val).astype('double')
    
    print("Start training the model")
    model.fit(data_conf)
    print("Finish training the model")
    if save_model:
        model.save("implicit_als_model/model")
        print("Saved the model")
