from preprocesser import generate_dataset
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, GBTRegressor, RandomForestRegressor
from pyspark_knn.ml.regression import KNNRegression

def predict(algorithm, paramGrid):
    global training
    global testing
    pipeline = Pipeline(stages=[algorithm])
    crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=RegressionEvaluator(labelCol='positive_rating_ratio'),
                            numFolds=2)#TODO:5 folds
    # Run cross-validation, and choose the best set of parameters.
    model = crossval.fit(training)
    # if isinstance(algorithm, LinearRegression):
    #     print("Coefficients: " + str(model.weights))
    if isinstance(algorithm, LinearRegression):
        print("linear regressor coefficients: " + str(model.bestModel.stages[-1].coefficients))
    #make prediction
    predictions =model.transform(testing)
    #print prediction samples
    predictions.select("features", "positive_rating_ratio", "prediction").show(5)
    #evaluate rmse
    evaluator = RegressionEvaluator(predictionCol="prediction", \
                    labelCol="positive_rating_ratio",metricName="rmse")
    print("rmse on test data = %g" % evaluator.evaluate(predictions))
training, testing = generate_dataset()

#linear regression
print("------------------------Linear Regression------------------------")
lr = LinearRegression(featuresCol='features', labelCol='positive_rating_ratio', maxIter=10)
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.5, 0.3, 0.1, 0.05, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.3, 0.5]).build()#TODO:add more param for elastic
predict(lr, paramGrid)

#decision tree
print("---------------------Decision Tree Regression---------------------")
dt = DecisionTreeRegressor(featuresCol="features",labelCol='positive_rating_ratio')
paramGrid=ParamGridBuilder().addGrid(dt.maxDepth, [5, 10, 30]) \
    .addGrid(dt.minInstancesPerNode, [1, 5, 10]).build()

predict(dt, paramGrid)

#random forest regression
print("---------------------Random Forest Regression---------------------")
rf = RandomForestRegressor(featuresCol="features",labelCol='positive_rating_ratio')
# Chain indexer and forest in a Pipeline
paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10, 30]) \
    .addGrid(rf.numTrees, [10, 20, 30, 50]).build()
predict(rf, paramGrid)

# Train a GBT model.
print("---------------------Gradient-boosted Tree Regression---------------------")
gbt = GBTRegressor(featuresCol="features",labelCol='positive_rating_ratio')
# Chain indexer and GBT in a Pipeline
paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [5, 10, 30]) \
    .addGrid(gbt.maxIter, [10, 20, 30]).build()
predict(gbt, paramGrid)

#KNN
#这部分的setup还有问题
print("------------------------------------KNN-----------------------------------")
knn = KNNRegression(featuresCol="features", labelCol="positive_rating_ratio")
paramGrid = ParamGridBuilder().addGrid(knn.topTreeSize, [1000, 2000]).build()
predict(knn, paramGrid)