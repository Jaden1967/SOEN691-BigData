from pyspark.ml import Pipeline
from pyspark.shell import sqlContext, sc
from pyspark.sql import *
from pyspark.sql.functions import udf, col
from datetime import date
from pyspark.sql.types import StringType, StringType, FloatType, IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoderEstimator, MinMaxScaler
import csv

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.model_selection import train_test_split
import mglearn
from math import sqrt

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
  return preprocess(training) ,preprocess(testing)


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
    df_new=df_new.select('features','positive_rating_ratio')
    return df_new


def predict(tr,te,pipeline, paramGrid,linear):
    crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=RegressionEvaluator(labelCol='positive_rating_ratio'),
                            numFolds=5)#TODO:5 folds
    # Run cross-validation, and choose the best set of parameters.
    model = crossval.fit(tr)
    if(linear):
        print("Model coefficients: " + str(model.bestModel.stages[-1].coefficients))
    else:
        print("Model coefficients: ", model.bestModel.stages[1])
    #make prediction
    predictions =model.transform(te)
    #print prediction samples
    predictions.select("features", "positive_rating_ratio", "prediction").show(10)
    #evaluate rmse
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="positive_rating_ratio",metricName="rmse")
    print("Root Mean Squared Error (RMSE) on test data = %g" % evaluator.evaluate(predictions))

# Linear Regression
def linear(tr,te):
    lr = LinearRegression(featuresCol='features', labelCol='positive_rating_ratio', maxIter=10)
    paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.5, 0.3, 0.1, 0.05, 0.01]) \
        .addGrid(lr.elasticNetParam, [0.3, 0.5]).build()  # TODO:add more param for elastic
    print("---------------------Linear Regression---------------------")
    pipeline = Pipeline(stages=[lr])
    predict(tr, te, pipeline, paramGrid, True)

# Random Forest Regression
def RF_Algorithm(tr,te,featureIndexer):
    rf = RandomForestRegressor(featuresCol="indexedFeatures",labelCol='positive_rating_ratio')
    # Chain indexer and forest in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, rf])
    print("---------------------Random Forest Regression---------------------")
    paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10, 15]) \
        .addGrid(rf.numTrees, [10, 20, 30, 50]).build()
    predict(tr, te, pipeline, paramGrid, False)

# Decision Tree Regression
def DT_Algorithm(tr,te,featureIndexer):
    # Train a DecisionTree model.
    dt = DecisionTreeRegressor(featuresCol="indexedFeatures",labelCol='positive_rating_ratio')
    # Chain indexer and tree in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, dt])
    paramGrid=ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 15]) \
        .addGrid(dt.minInstancesPerNode, [1, 5, 10]).build()
    print("---------------------Decision Tree Regression---------------------")
    predict(tr, te, pipeline, paramGrid, False)

# Gradient-boosted tree regression
def gradient_Boosted(tr,te,featureIndexer):
    # Train a GBT model.
    gbt = GBTRegressor(featuresCol="indexedFeatures",labelCol='positive_rating_ratio', maxIter=10)
    # Chain indexer and GBT in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, gbt])
    print("---------------------Gradient-boosted Tree Regression---------------------")
    paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [5, 10, 15]) \
       .addGrid(gbt.maxIter, [10, 20, 30]).build()
    predict(tr, te, pipeline, paramGrid, False)



originalDataPath = './data/new_steam.csv'
unwantedCols = ['appid', 'name', 'publisher', 'genres', 'categories']
spark = init_spark()
data = spark.read.option("quote", "\"").option("escape", "\"").csv(originalDataPath, header=True)
newData = preprocess(data)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(newData)
training, testing = newData.randomSplit([0.9, 0.1])

linear(training,testing)
RF_Algorithm(training,testing,featureIndexer)
DT_Algorithm(training,testing,featureIndexer)
gradient_Boosted(training,testing,featureIndexer)