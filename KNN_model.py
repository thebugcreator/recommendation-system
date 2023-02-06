import pandas as pd
from surprise import Dataset,Reader
from surprise.model_selection import cross_validate
from surprise import KNNBasic,KNNWithMeans,accuracy
from surprise.model_selection import train_test_split





df = pd.read_csv("data/goodreads_interactions.csv", sep=",", nrows=10_000)
df = df[["user_id","book_id", "rating"]].rename(columns={"user_id":"userID",
                                    "book_id":"itemID","rating":"rating"})
                                    
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df, reader)

trainset, testset=train_test_split(data, test_size=0.2)
#item-based (Cosine)
sim_options = {
    "user_based": False,  # compute  similarities between users
    "name":"cosine"
}
algo = KNNBasic(sim_options=sim_options,k=40, min_k=5) #default value of k is 40
# 
#cv=cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, return_train_measures=True, verbose=True)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
accuracy.mae(predictions)
