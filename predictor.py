from preprocesser import generate_dataset

def reFormat(df):
    df = df.drop(df.columns.difference(['features', 'positive_rating_ratio']), 1, inplace=True)
    df = df.withColumnRenamed('positive_rating_ratio', 'label')
    return df

training, testing = generate_dataset()
training = reFormat(training)
testing = reFormat(testing)
