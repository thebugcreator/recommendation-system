from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark RegSys example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

ratings = spark.read.option("header", True).option("inferSchema", True).csv("datasets/goodreads_interactions.csv")
(training, test) = ratings.randomSplit([0.8, 0.2])

als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="book_id", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each book
bookRecs = model.recommendForAllItems(10)

# Generate top 10 movie recommendations for a specified set of users
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
# Generate top 10 user recommendations for a specified set of books
books = ratings.select(als.getItemCol()).distinct().limit(3)
booksSubSetRecs = model.recommendForItemSubset(books, 10)
