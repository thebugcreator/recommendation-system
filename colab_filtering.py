from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark RecSys") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

ratings = spark.read.option("header", True).csv("datasets/goodreads_interactions_sample.csv")
(training, test) = ratings.randomSplit([0.8, 0.2])


