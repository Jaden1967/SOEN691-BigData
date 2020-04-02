from preprocesser import generate_dataset
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression

def predict(algorithm, paramGrid):
    global training
    global testing
    pipeline = Pipeline(stages=[algorithm])
    crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=RegressionEvaluator(),
                            numFolds=5)#5 folds
    # Run cross-validation, and choose the best set of parameters.
    model = crossval.fit(training)
    print("Coefficients: " + str(model.coefficients))

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
lr = LinearRegression(featuresCol='features', labelCol='positive_rating_ratio', maxIter=10)
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.5, 0.3, 0.1, 0.05, 0.01]) \
    .addGrid(lr.elasticNetParam, [0.3, 0.8]) \
    .build()
predict(lr, paramGrid)

#random forest regression

#KNN regression

