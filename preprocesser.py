
from pyspark.sql import *
from pyspark.sql.functions import udf, countDistinct
from datetime import date
from pyspark.sql.types import StringType
from pyspark.ml.feature import OneHotEncoder, StringIndexer

def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark
def getDiffDays(startDate):#get diff days since release to 2019-05-15
    
    startDateList = map(lambda x : int(x), startDate.split('-'))
    endDateList = [2019, 5, 15]
    startDate = date(*startDateList)
    endDate = date(*endDateList)
    diffDays = (endDate - startDate).days
    return str(diffDays)

def getPositiveRatingRatio(po, ne):
    return str(int(po)/(int(po) + int(ne)))
def getNumberOfOwners(owners):
    # try:
    #     li = map(lambda x : int(x), owners.split('-'))
    #     return str(int(sum(li)/2))
    # except ValueError:
    #     print('caught' + owners)
    #     exit(1)
    li = map(lambda x : int(x), owners.split('-'))
    return str(int(sum(li)/2))

def oneHotEncode(df, colName):
    indexColName = colName + 'index'
    vecColName = colName + 'Vec'
    stringIndexer = StringIndexer(inputCol=colName, outputCol=indexColName)
    model = stringIndexer.fit(df)
    indexed = model.transform(df)
    encoder = OneHotEncoder(inputCol=indexColName, outputCol=vecColName)
    encoder.setDropLast(False)
    encoded = encoder.transform(indexed)
    return encoded

originalDataPath = './data/steam.csv'
unwantedCols1 = ['appid', 'name', 'publisher', 'genres']
spark = init_spark()
df = spark.read.option("quote", "\"").option("escape", "\"").csv(originalDataPath, header=True)

#remove unwanted columns
for it in unwantedCols1:
    df = df.drop(it)
#add days column
udfGetDiffDays = udf(getDiffDays, StringType())
df = df.withColumn('days', udfGetDiffDays(df.release_date))
df = df.drop('release_date')
#add positive rating ratio column
# df = rdd.map(lambda x : x + (getPositiveRatingRatio(x.positive_ratings, x.negative_ratings),)).toDF(rawData.columns + ['positive_rating_ratio'])
udfGetPositiveRatingRatio = udf(getPositiveRatingRatio, StringType())
df = df.withColumn('positive_rating_ratio', udfGetPositiveRatingRatio(df.positive_ratings, df.negative_ratings))
df = df.drop('positive_ratings')
df = df.drop('negative_ratings')

#add number_of_owners column
# rdd = df.rdd
# df = rdd.map(lambda x : x + (getNumberOfOwners(x.owners),)).toDF(rawData.columns + ['number_of_owners'])
udfGetNumberOfOwners = udf(getNumberOfOwners, StringType())
df = df.withColumn('number_of_owners', udfGetNumberOfOwners(df.owners))
df = df.drop('owners')
# df.agg(countDistinct("developer")).show()#17113 developers
encodedDf = oneHotEncode(df, 'steamspy_tags')
encodedDf.show()





