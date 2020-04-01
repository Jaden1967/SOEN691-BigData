from pyspark.ml import Pipeline
from pyspark.shell import sqlContext, sc
from pyspark.sql import *
from pyspark.sql.functions import udf
from datetime import date
from pyspark.sql.types import StringType, StringType, FloatType
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoderEstimator, MinMaxScaler
import csv


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
def generateNewRawData(df):
    
    def getNumberOfGames(developer):
        return str(gameFrequency[developer])
    rdd = df.rdd
    tup = rdd.groupBy(lambda x : x[4]).collect()
    gameFrequency = {}
    for i in tup:
        gameFrequency[i[0]] = len(i[1])

    
    #add positive rating ratio column
    # df = rdd.map(lambda x : x + (getPositiveRatingRatio(x.positive_ratings, x.negative_ratings),)).toDF(rawData.columns + ['positive_rating_ratio'])
    udfGetNumberOfGames = udf(getNumberOfGames, StringType())
    df = df.withColumn('developer_products', udfGetNumberOfGames(df.developer))
    df = df.drop('developer')
    df.show()
    df.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save('./data/new_steam.csv')
def generateTagSet(df):
    tagSet = []
    tags_list = df.rdd.map(lambda x : x.steamspy_tags).collect()
    print(tags_list)
    for tags in tags_list:
        for tag in tags.split(';'):
            if tag not in tagSet:
                tagSet.append(tag)

    print(tagSet)
    with open('./data/steamspy_tags.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(tagSet)

def isintheset(all_label,string):
    stringlist = all_label.split(";")
    if string in stringlist:
      return 1
    else:
      return 0
def generateCatColumns(df, li, col):
    iterCat = None
    def isInTheSet(all_label):
        stringlist=all_label.split(";")
        if iterCat in stringlist:
            return 1
        else:
            return 0
    udfIsInTheSet = udf(isInTheSet, StringType())
    
    for cat in li:
        iterCat = cat
        print(iterCat)
        df = df.withColumn(cat, udfIsInTheSet(col))
    return df
def xiyun(df):
    with open('./data/steamspy_tags.csv') as file:
         readCSV = csv.reader(file, delimiter=',')
         for row in readCSV:
             tag_set=row

    df = generateCatColumns(df, tag_set, df.steamspy_tags)
    df = generateCatColumns(df, ['windows','mac','linux'], df.platforms)
    df.show()

def generate_dataset():
  global training
  global testing

#   for i in range(0,2):
#      if i==0:
#          dataframe=training
#      else:
#          dataframe=testing
#     #get all feature[]
#      with open('./data/steamspy_tags.csv') as file:
#          readCSV = csv.reader(file, delimiter=',')
#          for row in readCSV:
#              tag_set=row

#      plat_forms=['windows','mac','linux']
#      tempodf=[]

#      #one hot encoder manually
#      for feature in tag_set:
#          for i in range(0,dataframe.count()):
#              tempodf.append(feature)
#          rdd1 = sc.parallelize(tempodf)
#          row_rdd = rdd1.map(lambda x: Row(x)).zipWithIndex()
#          df = sqlContext.createDataFrame(row_rdd, [feature])

#          dataframe=dataframe.join(df, df.columns[-1]==dataframe.columns[-1], how='right')
#          tempodf.clear()

#      for plat_form in plat_forms:
#          for i in range(0,dataframe.count()):
#              tempodf.append(plat_form)
#          rdd1 = sc.parallelize(tempodf)
#          row_rdd = rdd1.map(lambda x: Row(x))
#          df = sqlContext.createDataFrame(row_rdd, [plat_form])
#          dataframe=dataframe.withColumn(plat_form,df.select(plat_form))
#          tempodf.clear()


#      udfisIndataset = udf(isintheset, StringType())

#      for feature in tag_set:

#         dataframe = dataframe.withColumn(feature+'_encoder', udfisIndataset(dataframe.steamspy_tags, dataframe.feature))
#         dataframe = dataframe.drop(feature)


#      for plat_form in plat_forms:
#          dataframe = dataframe.withColumn(plat_form + '_encoder',udfisIndataset(dataframe.platforms, dataframe.platform))
#          dataframe = dataframe.drop(plat_form)


#      #######pipline: add feature vector#####

#      all_features=dataframe.schema.names
#      Assember= VectorAssembler(
#              inputCols=all_features,
#              outputCol="all_features")
#      pipline=Pipeline(stages=all_features+[Assember])
#      if i ==0:
#       training_df= pipline.fit(dataframe).transform(dataframe)
#      else:
#       testing_df= pipline.fit(dataframe).transform(dataframe)

  return preprocess(training) ,preprocess(testing)


# generateNewRawData(df)
# generateTagSet(df)

def preprocess(df):
    global unwantedCols
    #remove unwanted columns
    for it in unwantedCols:
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
    xiyun(df)


    # all_features=df.schema.names
    # all_string_features=['developer','platforms','categories','steamspy_tags']
    # all_int_features=['english','required_age','achievements', 'average_playtime', 'median_playtime', 'days',  'number_of_owners']
    # all_float_features=['price']
    # for column in all_int_features:
    #     df=df.withColumn(column,df[column].cast(StringType()))
    # for column in all_float_features:
    #     df=df.withColumn(column,df[column].cast(FloatType()))
    # #one hot encoding df category

    # def one_hot(dataframe):

    #     indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in all_string_features]
    #     encoder = OneHotEncoderEstimator(
    #         inputCols=[indexer.getOutputCol() for indexer in indexers],
    #         outputCols=["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers]
    #     )
    #     assembler = VectorAssembler(
    #         inputCols=encoder.getOutputCols(),
    #         outputCol="cat_features"
    #     )
    #     # combine all the numberical_feature togeher
    #     assembler2=VectorAssembler(
    #         inputCols=all_int_features+all_float_features,
    #         outputCol="num_features"
    #     )
    #     pipeline = Pipeline(stages=indexers+[encoder,assembler,assembler2])
    #     df_r = pipeline.fit(dataframe).transform(dataframe)

    #     # scaler = MinMaxScaler(inputCol="num_features", outputCol="scaled_Num_Features")
    #     # # Compute summary statistics and generate MinMaxScalerModel
    #     # scalerModel = scaler.fit(df_r)
    #     # scaledData = scalerModel.transform(df_r)
    #     # # print(scaledData.count())
    #     return df_r



    # new_df=one_hot(df)
    # new_df.show()
    # print(new_df.schema.names)
    # print(new_df.first()[22])

originalDataPath = './data/new_steam.csv'
unwantedCols = ['appid', 'name', 'publisher', 'genres', 'categories']
spark = init_spark()
training, testing = spark.read.option("quote", "\"").option("escape", "\"").csv(originalDataPath, header=True).randomSplit([0.9, 0.1])
generate_dataset()