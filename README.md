# Clustering-Customer-Demand

Business Objective - Forecast the electricity usage for an individual customer for the next year(2018) based on the 2 years of data(2016 and 2017). This model can then be applied to all the customers on scale and Power companies can use this data to decide on-peak and off-peak usages and for better production planning.

Clustering is an unsupervised machine learning method. Here I have used clustering to cluster the electricity usage of a customer based on his previous usage history. Here we are clustering the time period into clusters. The entire day(24 hours) is divided into 24 different periods and the data for electricity usage for each hour is available to us. 

I have used Kmeans algorithm with k = 6.

Why kmeans ? 
Based on the scatter plots we can see that the data seems to be separable into clusters and we have only 2 numeric columns as inputs for the model. It is also the simplest one. 

Why 6 clusters ? 
We need to divide 24 periods in a day into clusters, and based on the scatter plot it was apparant that atleast 3 clusters will be there to get better clustering output. Also we cannot have large number of clusters as it will make the study more cumbersome. SO optimal number of clusters was decided. 


The file dict_holi and dict_week contains the electricity usage data for holiday/weekend and weekday.
The file kwh_byhour is the main data file. 
The file kwh_byhour_calculation is the file which contains calculations of demand factor.
