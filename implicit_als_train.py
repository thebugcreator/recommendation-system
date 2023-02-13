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
    parser.add_argument("--save_model", type=bool, default=False, help="Decide whether or not to save the model")
    parser.add_argument("--metric", type=str, default="rmse", help="Evaluation metrics")
    args = parser.parse_args()
    
    csv_path = args.csv_path
    save_model = args.save_model
    metric = args.metric
    
    # Load the interaction dataset
    df = pd.read_csv(csv_path)
    
    # Start creating the confidence matrix
    df["confidence"] = [calc_confidence(row[["is_read","rating","is_reviewed"]]) for index, row in df.iterrows()]
    
    data = df[["user_id", "book_id", "confidence"]]
    
    sparse_item_user = sparse.csr_matrix((data['confidence'].astype(float), (data['book_id'], data['user_id'])))
    sparse_user_item = sparse.csr_matrix((data['confidence'].astype(float), (data['user_id'], data['book_id'])))
    
    # Building the model
    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)
    alpha_val = 40
    data_conf = (sparse_item_user * alpha_val).astype('double')
    model.fit(data_conf)

