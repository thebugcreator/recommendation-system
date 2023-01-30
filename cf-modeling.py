import pandas as pd
from lightfm import LightFM
from scipy.sparse import csr_matrix




df = pd.read_csv("data/goodreads_interactions.csv", sep=",", nrows=100_000)
df = df[["user_id","book_id", "rating"]]

# fil na with 0 after pivoting
user_mat = df.pivot_table(index="user_id",columns="book_id", values="rating").fillna(0)

# use scipy csr matrix
user_mat_csr = csr_matrix(user_mat.values)

# Use LightFM
model = LightFM(loss='warp',
                random_state=2016,
                learning_rate=0.90,
                no_components=150,
                user_alpha=0.000005)
model = model.fit(user_mat_csr,
                  epochs=100,
                  num_threads=16, verbose=False)
