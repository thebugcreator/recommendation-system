from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from pyspark.sql import SparkSession

import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Colaborative Filtering Model with Spark")
    parser.add_argument("--csv_path", type=str, default="datasets/goodreads_interactions.csv", help="Path to the csv interaction file.")
    parser.add_argument("--save_model", type=bool, default=False, help="Decide whether or not to save the model")
    parser.add_argument("--metric", type=str, default="rmse", help="Evaluation metrics")
    parser.add_argument("--max_iter", type=int, default=10, help="Model max iteration")
    parser.add_argument("--reg_param", type=float, default=0.01, help="Model regParam")
    parser.add_argument("--bridge_col", type=str, default="rating", help="Interaction column")
    
    args = parser.parse_args()
    
    csv_path = args.csv_path
    save_model = args.save_model
    metric = args.metric
    max_iter = args.max_iter
    reg_param = args.reg_param
    bridge_col = args.bridge_col
    
    
    spark = SparkSession \
        .builder \
        .appName("Python Spark RegSys example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    ratings = spark.read.option("header", True).option("inferSchema", True).csv(csv_path)
    (training, test) = ratings.randomSplit([0.8, 0.2])

    als = ALS(maxIter=max_iter, regParam=reg_param, userCol="user_id", itemCol="book_id", ratingCol=bridge_col,
              coldStartStrategy="drop")
    model = als.fit(training)

    # Evaluate the model
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName=metric, labelCol="rating",
                                    predictionCol="prediction")
    score = evaluator.evaluate(predictions)
    
    print("Evaluation score using {0} : {1}".format(metric, score))

    if save_model:
        model.save("als_models")

    # Generate top 10 movie recommendations for each user
    #userRecs = model.recommendForAllUsers(10)
    # Generate top 10 user recommendations for each book
    #bookRecs = model.recommendForAllItems(10)

    # Generate top 10 movie recommendations for a specified set of users
    #users = ratings.select(als.getUserCol()).distinct().limit(3)
    #userSubsetRecs = model.recommendForUserSubset(users, 10)
    # Generate top 10 user recommendations for a specified set of books
    #books = ratings.select(als.getItemCol()).distinct().limit(3)
    #booksSubSetRecs = model.recommendForItemSubset(books, 10)
