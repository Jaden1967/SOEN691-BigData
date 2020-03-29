
from pyspark.sql import *
from datetime import date


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark
def getDiffDays(startDate, endDate):
    print(startDate)
    startDateList = map(lambda x : int(x), startDate.split('-'))
    endDateList = map(lambda x: int(x), endDate.split('-'))
    startDate = date(*startDateList)
    endDate = date(*endDateList)
    diffDays = (endDate - startDate).days
    print('diff days: ' + str(diffDays))
    return diffDays

originalDataPath = './data/steam.csv'
spark = init_spark()
rawData = spark.read.csv(originalDataPath, header=True)
print(rawData.columns)

#remove unwanted columns
for it in ['appid', 'name', 'publisher', 'genre']:
    rawData = rawData.drop(it)
# rawData.printSchema()
#convert release date to diff days since release to 2019-05-15
rawData.select('release_date').rdd.flatMap(lambda x : (getDiffDays(x.release_date, '2019-05-15'), )).toDF()#这块我不会写，就是把release_date所有row换成diffdays
rawData.show()
