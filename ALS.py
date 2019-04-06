#!/usr/bin/env python
# coding: utf-8

# In[33]:


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.mllib.recommendation import MatrixFactorizationModel, Rating
import math
from pyspark.sql.types import IntegerType, FloatType


# In[2]:


conf= SparkConf()
#pattern = re.compile("\t+")
sc= SparkContext(conf=conf)


# In[3]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# In[4]:


rating_rdd = sc.textFile("file:///home/013731783/256_HW1/train.dat")
test_rdd = sc.textFile("file:///home/013731783/256_HW1/test.dat")


# In[5]:


ratings = rating_rdd.map(lambda l: l.split("\t")).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
test_data = test_rdd.map(lambda m: m.split("\t")).map(lambda m: Row(user=int(m[0]), product=int(m[1])))
ratings_df = spark.createDataFrame(ratings)
(training, test) = ratings_df.randomSplit([0.8, 0.2])
#validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
#test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))


# In[6]:


als = ALS(maxIter=8, regParam=0.087, userCol="user", itemCol="product", ratingCol="rating",
          coldStartStrategy="nan")


# In[7]:


model = als.fit(training)


# In[8]:


test_opt= test.withColumn("col_id", monotonically_increasing_id())
predictions_valid = model.transform(test_opt)


# In[ ]:





# In[9]:


pred= predictions_valid.sort(predictions_valid.col_id)
pred_tup= pred.select(pred.col_id,pred.product,pred.user,pred.prediction)


# In[10]:


pred_val = pred_tup.select("prediction").rdd.flatMap(list).take(2154)


# In[11]:


final_valid=[]
for x in pred_val:
    if math.isnan(x):
        final_valid.append(2.0)


# In[41]:


#evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
#                                predictionCol="value")


# In[42]:


#pred_val_df = spark.createDataFrame(final_valid,IntegerType())


# In[44]:


#rmse = evaluator.evaluate(pred_val_df)
#print("Root-mean-square error = " + str(rmse))


# In[14]:


test_data_df = spark.createDataFrame(test_data)
test_result = test_data_df.withColumn("col_id", monotonically_increasing_id())


# In[15]:


model = als.fit(ratings_df)


# In[16]:


predictions = model.transform(test_result)


# In[17]:


predictions


# In[18]:


pt= predictions.sort(predictions.col_id)
pred_out_tup= pt.select(pt.col_id,pt.product,pt.user,pt.prediction)


# In[19]:


pt


# In[20]:


pred_output = pred_out_tup.select("prediction").rdd.flatMap(list).take(2154)


# In[21]:


pred_output


# In[45]:


final=[]
for x in pred_output:
    if math.isnan(x):
        final.append(2)
    else:
        final.append(int(round(x, 0))) 


# In[46]:


final


# In[25]:


f = open("outputFile_22ndMar_v6.txt","w")
for f1 in final:
    f.write(str(f1)+"\n")
    
