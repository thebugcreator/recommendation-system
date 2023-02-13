import pandas as pd
import numpy as np
import scipy.sparse as sparse
import random
import implicit

model_path = "implicit_als_model/model.npz"

model = implicit.als.AlternatingLeastSquares(model_path)
sparse_book_user = sparse.load_npz("implicit_als_model/item_user.npz")
sparse_user_book = sparse.load_npz("implicit_als_model/user_item.npz")
