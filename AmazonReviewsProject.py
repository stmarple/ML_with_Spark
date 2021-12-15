
# In[Libraries]
from __future__ import print_function
import re
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import  split, col
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, SVMWithSGD
from pyspark.mllib.regression import LabeledPoint

# In[Spark Context]
try:
    sc = SparkContext()
except:
    sc.stop()
    sc = SparkContext()

# In[Functions]
def createKeyListOfWordsDF(df, column, label):
    if label == 'rating':
        dtype = 'float'
    else:
        dtype = 'string'
    filtered = (
        df.select(
            df[column].cast(dtype).alias(label),
            df._c21.cast('string').alias('review') )
    )
    if label == 'rating':
        filtered = filtered.filter(df._c19 != 3) # remove ratings = 3 and Null values

    cols = [label,'review']
    col_exprs = [split(col(x), " ").alias(x) for x in cols]
    KeyListOfWords = filtered.select(*col_exprs)
    return KeyListOfWords

def topWords(keyWordList, numberofwords):
    allCounts = keyWordList.flatMap(lambda x: x[1]).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
    topwords = allCounts.top(numberofwords, key=lambda x: x[1]) # top words
    return sc.parallelize(range(numberofwords)).map(lambda x : (topwords[x][0], x))

def topWordsCheck(dictionary, lst_inQuotes_sep_by_commas, index=0):
    lst = lst_inQuotes_sep_by_commas.split(',')
    return dictionary.filter(lambda x: x[index] in lst).collect()

def freqArray (listOfIndices, numberofwords):
    returnVal = np.zeros (numberofwords)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    returnVal = np.divide(returnVal, numberofwords)
    return returnVal

def createTF(dictionary, keyWordList, numberofwords):
    allDictionaryWords = (keyWordList.flatMap(lambda x: ((j, x[0]) for j in x[1])).join(dictionary) # dictionary words indexed
                          .map(lambda x: (x[1][0], x[1][1]))          # maps document with word frequency
                          .groupByKey().map(lambda x: (x[0], freqArray(x[1],numberofwords))))
    return allDictionaryWords

def xy_map_ratings(tf):   #  0 = negative rating, 1 = positive rating
    return tf.map(lambda x: ((1 if x[0] > 3.0 else 0), x[1]))  # y is a binary classification (1 or 0)

def xy_map_categories(tf, category):
    return tf.map(lambda x: ((1 if category in x[1] else 0), x[1]))

def evaluate_training_model(xy, labelsNPreds):
    TP = labelsNPreds.filter(lambda x: x[0]==1 and x[1]==1).count()
    TN = labelsNPreds.filter(lambda x: x[0]==0 and x[1]==0).count()
    FP = labelsNPreds.filter(lambda x: x[0]==1 and x[1]==0).count()
    FN = labelsNPreds.filter(lambda x: x[0]==0 and x[1]==1).count()

    try:
        trainErr = labelsNPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(xy.count())
    except:
        trainErr = 'undefined'
    try:
        precision = TP/(TP+FP)
    except:
        precision = 'undefined'
    try:
        recall = TP/(TP+FN)
    except:
        recall = 'undefined'
    try:
        F1 = (2 * precision * recall) / (precision + recall)
    except: F1 = 'undefined'

    print(f'\nTP: {TP}\nTN: {TN}\nFP: {FP}\nFN: {FN}')
    print(f'Precision: {precision}\nRecall: {recall}\nF-measure: {F1}\nTraining Error: {trainErr}\n')
    return TP, TN, FP, FN, precision, recall, F1, trainErr

def topValues(coef):
    reg = coef.copy()
    return np.argsort(-reg)[:5]

def getIndexedWords(word_indexes, dictionary):
    return dictionary.filter(lambda x: x[1] in word_indexes).map(lambda x: (x[0], x[1])).collect()

def build_model(xy, regression):
    parsedData = xy.map(lambda x: LabeledPoint(x[0], x[1]))
    if regression == 'logistic':
        model = LogisticRegressionWithLBFGS.train(parsedData)
    else:
        model = SVMWithSGD.train(parsedData, iterations=100)
    return parsedData.map(lambda p: (p.label, model.predict(p.features)))

# In[File path]
#file_location = 'E:/Boston_University/Grad_School/CS777/Project/data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv'

# Larger dataset
file_location = 'E:/Boston_University/Grad_School/CS777/Project/data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv'

data = sc.textFile(file_location, 1)


# In[Task 1: Ratings Vs Reviews]
spark = SparkSession.builder.getOrCreate()
tbl = spark.read.csv(file_location)

# In[]
filter_df = createKeyListOfWordsDF(tbl,'_c19', 'rating')
rdd = (filter_df.rdd.map(lambda x: (''.join(x[0]), [x.lower() for x in x[1]]) )).map(lambda x: (float(x[0]), x[1]))
num_words = 500  # I had to reduce this in order to get the tf to work.  Is it possible that all these reviews have < 5000 unique words?
keyAndListOfWords = rdd
rdd_dict = topWords( keyAndListOfWords, num_words)
tf = createTF(rdd_dict, keyAndListOfWords, num_words)
xyMap = xy_map_ratings(tf)

# In[Task 1 A: Logistic Regression With LBFGS (or Limited-Memory Broyden–Fletcher–Goldfarb–Shanno)]
labelsAndPreds = build_model(xyMap, 'logistic')
evaluate_training_model(xyMap, labelsAndPreds)

# In[Task 1 B: SVM With Stochastic Gradient Descent]
labelsAndPreds = build_model(xyMap, 'svm')
evaluate_training_model(xyMap, labelsAndPreds)


# In[Task 2: Categories Vs Reviews]
''' This analysis was a bit on the simple side so I'll test reviews against the category where:
    1 = home and garden and 0 for all others  in primaryCategories
'''
filter_df = createKeyListOfWordsDF(tbl,'_c7', 'primaryCategories')
rdd = filter_df.rdd.map(lambda x: (''.join(x[0]), [x.lower() for x in x[1]] ) )
keyAndListOfWords = rdd

# In[]
num_words = 3500
rdd_dict = topWords( keyAndListOfWords, num_words)
tf = createTF(rdd_dict, keyAndListOfWords, num_words)
xyMap = xy_map_categories(tf, 'Health & Beauty')

# In[Task 2: Logistic Regression]
labelsAndPreds = build_model(xyMap,'logistic')
evaluate_training_model(xyMap, labelsAndPreds)

''' This classification model doesn't seem to fit this data, I should try another model
    This data set just doesn't seem large enough to learn anything
'''
# In[Task 2: SVM]
labelsAndPreds = build_model(xyMap,'svm')
evaluate_training_model(xyMap, labelsAndPreds)











