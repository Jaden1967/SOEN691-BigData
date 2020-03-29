
from pyspark.sql import *
from pyspark.sql.functions import lit
from datetime import date


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark
def getDiffDays(startDate, endDate):
    
    startDateList = map(lambda x : int(x), startDate.split('-'))
    endDateList = map(lambda x: int(x), endDate.split('-'))
    startDate = date(*startDateList)
    endDate = date(*endDateList)
    diffDays = (endDate - startDate).days
    return str(diffDays)
    
    
    
originalDataPath = './data/steam.csv'
unwantedCols = ['appid', 'name', 'publisher', 'genres']
spark = init_spark()
rawData = spark.read.option("quote", "\"").option("escape", "\"").csv(originalDataPath, header=True)

#remove unwanted columns
for it in unwantedCols:
    rawData = rawData.drop(it)
rdd = rawData.rdd
#convert release date to diff days since release to 2019-05-15
df = rdd.map(lambda x : x + (getDiffDays(x.release_date, '2019-05-15'),)).toDF(rawData.columns + ['days'])
df = df.drop('release_date')
df.show()