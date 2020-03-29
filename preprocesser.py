
from pyspark.sql import *


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark
originalDataPath = './data/steam.csv'
spark = init_spark()
rawData = spark.read.csv(originalDataPath, header=True)
print(rawData.columns)

#remove unwanted columns
for it in ['appid', 'name', 'publisher', 'genre']:
    rawData = rawData.drop(it)
print(rawData.columns)