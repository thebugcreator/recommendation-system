from pyspark.sql import Row
from pyspark.sql import SparkSession

from pyspark.ml.recommendation import ALS, ALSModel

# Create a spark session
xmx = "16g"
spark = SparkSession \
    .builder \
    .appName("Python Spark RegSys example") \
    .config("spark.driver.memory", xmx) \
    .getOrCreate()

# TODO: Load the spark ALS model
model = ALSModel.load("als_model")
# TODO: Lazy loading the related dataframes
#df_users = spark.read.option("header", True).option("inferSchema", True).csv("datasets/goodreads_users.csv")
df_books = spark.read.option("header", True).option("inferSchema", True).csv("datasets/goodreads_books_sample.csv")


def recommend_books_for_each_user(n=10):
    """Generate top 10 movie recommendations for each user"""
    userRecs = model.recommendForAllUsers(n)
    return userRecs

def recommend_users_for_each_book(n=10):
    """Generate top 10 user recommendations for each book"""
    bookRecs = model.recommendForAllItems(10)
    return bookRecs
    

def recommend_books_for_specific_users(user_ids, n=10):
    """Generate top 10 movie recommendations for a specified set of users"""
    users = ratings.select(als.getUserCol()).distinct().limit(3)
    userSubsetRecs = model.recommendForUserSubset(users, n)
    return userSubsetRecs

def recommend_users_for_specific_books(book_ids, n=10):
    """Generate top 10 user recommendations for a specified set of books"""
    books = ratings.select(als.getItemCol()).distinct().limit(3)
    booksSubSetRecs = model.recommendForItemSubset(books, n)
    return booksSubSetRecs
