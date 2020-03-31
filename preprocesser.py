from pyspark.ml import Pipeline
from pyspark.sql import *
from pyspark.sql.functions import udf
from datetime import date
from pyspark.sql.types import StringType, IntegerType, FloatType
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoderEstimator, MinMaxScaler


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
def split_features(lines):
     feature_list=lines.split(";")

     return feature_list


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
df.show()
#add positive rating ratio column
# df = rdd.map(lambda x : x + (getPositiveRatingRatio(x.positive_ratings, x.negative_ratings),)).toDF(rawData.columns + ['positive_rating_ratio'])
udfGetPositiveRatingRatio = udf(getPositiveRatingRatio, StringType())
df = df.withColumn('positive_rating_ratio', udfGetPositiveRatingRatio(df.positive_ratings, df.negative_ratings))
df = df.drop('positive_ratings')
df = df.drop('negative_ratings')
df.show()

#add number_of_owners column
# rdd = df.rdd
# df = rdd.map(lambda x : x + (getNumberOfOwners(x.owners),)).toDF(rawData.columns + ['number_of_owners'])
udfGetNumberOfOwners = udf(getNumberOfOwners, StringType())
df = df.withColumn('number_of_owners', udfGetNumberOfOwners(df.owners))
df = df.drop('owners')
df.show()


all_features=df.schema.names
all_string_features=['developer','platforms','categories','steamspy_tags']
all_int_features=['english','required_age','achievements', 'average_playtime', 'median_playtime', 'days',  'number_of_owners']
all_float_features=['price']
for column in all_int_features:
    df=df.withColumn(column,df[column].cast(IntegerType()))
for column in all_float_features:
    df=df.withColumn(column,df[column].cast(FloatType()))
#one hot encoding df category

def one_hot(dataframe):

    indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in all_string_features]
    encoder = OneHotEncoderEstimator(
        inputCols=[indexer.getOutputCol() for indexer in indexers],
        outputCols=["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers]
    )
    assembler = VectorAssembler(
        inputCols=encoder.getOutputCols(),
        outputCol="cat_features"
    )
    # combine all the numberical_feature togeher
    assembler2=VectorAssembler(
        inputCols=all_int_features+all_float_features,
        outputCol="num_features"
    )
    pipeline = Pipeline(stages=indexers+[encoder,assembler,assembler2])
    df_r = pipeline.fit(dataframe).transform(dataframe)

    # scaler = MinMaxScaler(inputCol="num_features", outputCol="scaled_Num_Features")
    # # Compute summary statistics and generate MinMaxScalerModel
    # scalerModel = scaler.fit(df_r)
    # scaledData = scalerModel.transform(df_r)
    # # print(scaledData.count())
    return df_r



new_df=one_hot(df)
new_df.show()
print(new_df.schema.names)
print(new_df.first()[22])