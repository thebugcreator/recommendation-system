import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise import KNNBasic, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split


df = pd.read_csv("goodreads_interactions.csv")

item = df["book_id"]
user = df["user_id"]
rating = df["rating"]

ratings_dict = {"item":item, "user":user, "rating":rating}

ratings = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings, reader)

train, test = train_test_split(data, test_size=0.2)

sim_options = {
    "user_based": False,  # compute  similarities between users
    "name":"cosine"
}
knn = KNNBasic(sim_options=sim_options,k=50, min_k=5) #default value of k is 40
# 
#cv=cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, return_train_measures=True, verbose=True)
knn.fit(train)
preds = algo.test(test)
acc_rmse = accuracy.rmse(preds)
acc_mae = accuracy.mae(preds)
