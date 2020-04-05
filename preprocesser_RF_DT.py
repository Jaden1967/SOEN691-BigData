from pyspark.ml import Pipeline
from pyspark.shell import sqlContext, sc
from pyspark.sql import *
from pyspark.sql.functions import udf, col
from datetime import date
from pyspark.sql.types import StringType, StringType, FloatType, IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoderEstimator, MinMaxScaler
import csv

from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import DecisionTreeRegressor

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
    # df.show()
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
def generateCatColumns(df, li, colName):
    # iterCat = None
    def isInTheSet(all_label, cat):
        stringlist=all_label.split(";")
        # print(iterCat)
        if cat in stringlist:
            return '1'
        else:
            return '0'
    def udf_helper(label_list):
        return udf(lambda l: isInTheSet(l, label_list))
    
    
    for cat in li:
        # iterCat = cat
        # print(iterCat)
        # df = df.withColumn(cat, udfIsInTheSet(df[col]))
        df = df.withColumn(cat, udf_helper(cat)(col(colName)))
    return df
def xiyun(df):
    with open('./data/steamspy_tags.csv') as file:
         readCSV = csv.reader(file, delimiter=',')
         for row in readCSV:
             tag_set=row
    df = generateCatColumns(df, tag_set, 'steamspy_tags')
    df = generateCatColumns(df, ['windows','mac','linux'], 'platforms')
    return df

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
    # df.show()
    #add positive rating ratio column
    # df = rdd.map(lambda x : x + (getPositiveRatingRatio(x.positive_ratings, x.negative_ratings),)).toDF(rawData.columns + ['positive_rating_ratio'])
    udfGetPositiveRatingRatio = udf(getPositiveRatingRatio, StringType())
    df = df.withColumn('positive_rating_ratio', udfGetPositiveRatingRatio(df.positive_ratings, df.negative_ratings))
    df = df.drop('positive_ratings')
    df = df.drop('negative_ratings')
    # df.show()

    #add number_of_owners column
    # rdd = df.rdd
    # df = rdd.map(lambda x : x + (getNumberOfOwners(x.owners),)).toDF(rawData.columns + ['number_of_owners'])
    udfGetNumberOfOwners = udf(getNumberOfOwners, StringType())
    df = df.withColumn('number_of_owners', udfGetNumberOfOwners(df.owners))
    df = df.drop('owners')
    # df.show()
    #########one hot manually####
    df=xiyun(df)
    # df.show()
    ###all feature to number###
    with open('./data/steamspy_tags.csv') as file:
         readCSV = csv.reader(file, delimiter=',')
         for row in readCSV:
             tag_set=row
    ##2。5D 有问题
    # tag_set.remove('2.5D')
    udfcol=['mac','windows','linux']+tag_set+['days','number_of_owners']
    original_int_features = ['english', 'required_age', 'achievements', 'average_playtime', 'median_playtime']
    all_float_features = ['price']
    for col in udfcol+original_int_features:
        df = df.withColumn(col, df[col].cast(IntegerType()))
    df=df.withColumn('positive_rating_ratio',df['positive_rating_ratio'].cast(FloatType()))
    df=df.withColumn('price',df['price'].cast(FloatType()))
    assember=VectorAssembler(inputCols=udfcol+original_int_features+all_float_features,
          outputCol="features")
    pipline=Pipeline(stages=[assember])
    df_new=pipline.fit(df).transform(df)
    # df_new.show()
    df_new=df_new.select('features','positive_rating_ratio')
    # df_new.show()
    return df_new

    # all_features=df.schema.names
    # all_string_features=['developer','platforms','categories','steamspy_tags']

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
data = spark.read.option("quote", "\"").option("escape", "\"").csv(originalDataPath, header=True)
# training, testing=data.randomSplit([0.9, 0.1])
# # training.show(10)
#
#
# tr,te=generate_dataset()
# tr.take(5)
# tr.show(5)


def RF_Algorithm(data):
    global training
    global testing
    newData=preprocess(data)
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(newData)

    training, testing = data.randomSplit([0.9, 0.1])
    tr, te = generate_dataset()

    rf = RandomForestRegressor(featuresCol="features",labelCol='positive_rating_ratio')
    # Chain indexer and forest in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, rf])
    # Train model.  This also runs the indexer.
    model = pipeline.fit(tr)

    # paramGrid = ParamGridBuilder().addGrid(rf.regParam, [0.5, 0.3, 0.1, 0.05, 0.01]) \
    #     .addGrid(rf.elasticNetParam, [0.3, 0.5]).build()  # TODO:add more param for elastic
    # pipeline = Pipeline(stages=[featureIndexer])
    # crossval = CrossValidator(estimator=pipeline,
    #                             estimatorParamMaps=paramGrid,
    #                             evaluator=RandomForestRegressor(labelCol="positive_rating_ratio"),
    #                             numFolds=5)#TODO:5 folds
    #     # Run cross-validation, and choose the best set of parameters.
    # model = crossval.fit(tr)
    #     # if isinstance(algorithm, LinearRegression):
    #     #     print("Coefficients: " + str(model.weights))
    # print("Coefficients: " + str(model.bestModel.stages[-1].coefficients))

    # Make predictions.
    predictions = model.transform(te)

    # Select example rows to display.
    print("---------------------Random Forest---------------------")
    predictions.select("features", "positive_rating_ratio", "prediction").show(10)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(labelCol="positive_rating_ratio", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    rfModel = model.stages[1]
    print(rfModel)  # summary only

def DT_Algorithm(data):
    global training
    global testing
    newData = preprocess(data)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(newData)
    training, testing = data.randomSplit([0.9, 0.1])
    tr, te = generate_dataset()

    # Train a DecisionTree model.
    dt = DecisionTreeRegressor(featuresCol="features",labelCol='positive_rating_ratio')
    # Chain indexer and tree in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, dt])
    # Train model.  This also runs the indexer.
    model = pipeline.fit(tr)

    # Make predictions.
    predictions = model.transform(te)
    # Select example rows to display.
    print("---------------------Decision Tree---------------------")
    predictions.select("features", "positive_rating_ratio", "prediction").show(10)
    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(labelCol="positive_rating_ratio", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
    treeModel = model.stages[1]
    # summary only
    print(treeModel)

RF_Algorithm(data)
DT_Algorithm(data)