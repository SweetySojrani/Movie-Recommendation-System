
#Collaborative Filtering system#

#Introduction:#
The objective of this assignment is to allow develop collaborative filtering models that can predict the rating of a specific item from a specific user given a history of other ratings. 

#Approach:#
As the data has only rating of the movies, we don’t have the details of movie genre, text, cast information,which is why we cannot use content based filtering. Collaborative filtering will be used because it is based on using the aggregated behaviour of large number of users to suggest relevant items to specific users. So, collaborative filtering method can be used to predict movie rating a particular user will give. So it can recommend the movies accordingly for eg, recommend 3+ rating movies predicted to users who have not watched the movie. From the collaborative filtering methods we have used explicit ALS method since we have movie ratings in our training set. 

#Implement Collaborative Filtering in pyspark:#
Convert the training and test file to RDD.
The training file has 3 columns of userId, movieId, rating separated by tab. I used map function to split the fields by ‘\t’ and create tuples of userId, movieId, rating.
Split the training RDD into training RDD, Validation RDD in the ratio of 0.8:0.2
The Validation RDD was mapped into tuples of UserId and MovieId. I removed the ratings so that it will be predicted by the ALS algorithm
I user ALS (Alternating least squares) 
I used parameters for ALS algorithm and tried changing the values to decide the optimized values for the parameters of the model
I found that maxteration = 20, regParam=0.1 gave the best RMSE with decent time.
Once the parameters for the ALS algorithm were finalized, I trained the model again with the complete training RDD. 
I added a column of col_id to the test RDD by using sql function of monotonically_increasing_id() which is increasing and unique, but not consecutive. Other option that I tried were RDD.zipWithUniqueId. But I went ahead with monotonically_increasing_id() because, the spark model.transform function changes the order of the prediction output. 
The complete model training was then used to predict the test data ratings.
I sorted the output by col_id, which is why we used monotonically_increasing_id() in order to handle the order change of the records after transform.
Then,the sorted prediction output was in a tuple of userid, movieid, rating, col_id.
I selected the prediction from the tuples and checked for nan values in the predictions. 
The prediction output had 3 values of nan since there are 3 users in the test file which do not exist in the training file. This is a scenario of slow start where a new user joins and has not rated any movie. In that case the movie recommendation system selects all the movies for recommendation which has rating above a certain threshold or average of all movies. Average will take a lot of time and since we are not considering any relationship between users or between movies. Considering a threshold makes more sense. I have considered threshold of 2 for these 3 users.

#Methodology of choosing parameters:# 
After running mutiple combinations of MaxIteration = 0 to 40. The Error rate scores were plotted for the Validation data, training the ALS each time with different set of parameters on training data. The least RMSE score was calculated and considered finally for predicting the rating on test data. Below is the graph plotted for MaxIteration parameters. Based on below graph , I chose Maxiteration = 20



Run the code: 
The libraries required for running the code are spark libraries of RegressionEvaluator,  ALS, SparkConf, SparkContext and sql function libraries of Row, monotonically_increasing_id. 
The code was run using below command of slurm by editing the python name and arguments in the slurm.sh file. File was run using below steps:
 - module load python3 
 - export SPARK_HOME=/opt/ohpc/pub/libs/spark
 - SPARK_HOME/bin/pyspark  
 - spark-slurm.sh

The commands resulted in an output file the ratings of each user,movie in the test file.
