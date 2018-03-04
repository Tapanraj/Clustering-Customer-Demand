# importing required libraries
import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
from datetime import datetime

# define a random seed value for np as we don't want values of any random variable to change as we run the script multiple times
np.random.seed(0)

# read the data into dataframe df
df = pd.read_excel(r'C:\Users\DELL\Desktop\Analytical-Decision-Modeling-2\519-case\kwh_byhour.xlsx', sheetname = 'kWh usage by hour')
print(df.columns)

# deleting Column named "Time in 12" as we dont require it for clustering
del df['Time in 12']

# we don't require data for 2015 as it is not complete and will bias our results for few of the months
df = df.drop(df[df.Date < '1/1/2016'].index)
print(df)

# The demand factor is a factor by which a month's demand is greater than the average monthly demand of electricity
demand_factor = [1.1858, 0.6007, 0.5395, 0.6469, 0.7751, 1.4129, 1.6577, 1.8551, 1.2478, 0.9865, 0.5119, 0.5802]

# entering the conditions to assign a value of demand factor for each row in the dataframe
conditions = [(df['Month'] == 1),(df['Month'] == 2),(df['Month'] == 3), (df['Month'] == 4), (df['Month'] == 5),
    (df['Month'] == 6), (df['Month'] == 7), (df['Month'] == 8), (df['Month'] == 9), (df['Month'] == 10),
    (df['Month'] == 11), (df['Month'] == 12)]

# adding a new column to dataframe named factor for entering demand factor
df['factor'] = np.select(conditions, demand_factor)
print(df)

# deseasonalize the demand so that the effect of the overall seasonality for a month doesn't affect the clustering of the period in a day
df['Deseasonalized_demand'] = df['Demand']/df['factor']
print(df)

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# assigning True if a day is U.S Federal Holiday, otherwise False
cal = calendar()
holidays = cal.holidays(start=df['Date'].min(), end=df['Date'].max())
df['Holiday'] = df['Date'].isin(holidays)
print (df)
print(holidays)
print(df['Holiday'].value_counts())

# Assign True if it is weekend, otherwise just copy the value as it is in the Holiday column
df['Weekend/Holiday'] = np.where((df['Day of the Week'] == 'Saturday') | (df['Day of the Week'] == 'Sunday'), 'True',df['Holiday'])
print(df)
print(df['Weekend/Holiday'].value_counts())

# delete the columns which are not required for clustering 
del df['Day of the Week']
del df['Holiday']
del df['Date']
del df['Month']
del df['Demand']
del df['factor']
print(df)

# Segregating the rows in dataframe according to weekend/holiday and putting it in a different dataframe clust_week if it is a weekday otherwise clust_holi
clust_week = df.loc[df['Weekend/Holiday'] == 'False']
del clust_week['Weekend/Holiday']
print(clust_week)

clust_holi = df.loc[df['Weekend/Holiday'] == 'True']
del clust_holi['Weekend/Holiday']
print(clust_holi)

print(clust_holi.dtypes)
print(clust_week.dtypes)

# replace null values with NaN so that we can impute it with mean in next steps
clust_holi.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
clust_week.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

# count number of null values and display it
print(clust_week.isnull().sum())
print(clust_holi.isnull().sum())
null_data_week = clust_week[clust_week.isnull().any(axis=1)]
null_data_holi = clust_holi[clust_holi.isnull().any(axis=1)]
print(null_data_week)
print(null_data_holi)

# Fill the Null values with mean values of the whole deseasonalized demand, so that it doesn't affect the clustering 
clust_week['Deseasonalized_demand'].fillna((clust_week['Deseasonalized_demand'].mean()), inplace=True)
clust_holi['Deseasonalized_demand'].fillna((clust_holi['Deseasonalized_demand'].mean()), inplace=True)       

print(clust_week.isnull().sum())
print(clust_holi.isnull().sum())

# Now run the Kmeans algorithm to make clusters
from sklearn.cluster import KMeans
kmeans_week = KMeans(n_clusters=6)
kmeans_holi = KMeans(n_clusters=6)
print(kmeans_week.fit(clust_week))
print(kmeans_holi.fit(clust_holi))
print()

#label1 = kmeans_week.predict(clust_week)
#label2 = kmeans_holi.predict(clust_holi)
#print(len(label1))
#print(len(label2))

# Either run the above commented lines or the below ones, both does the same thing
labels_week = kmeans_week.labels_
labels_holi = kmeans_holi.labels_
print(labels_week)
print(labels_holi)

# Enter the column in dataframe for labels
clust_week['Labels'] = labels_week
clust_holi['Labels'] = labels_holi

print(clust_week)
print(clust_holi)

# The below lines with 3 comment signs can be run to print the result as csv file
###clust_week.to_csv(r'C:\Users\DELL\Desktop\Analytical-Decision-Modeling-2\519-case\dict_week.csv', encoding='utf-8', index=False)
###clust_holi.to_csv(r'C:\Users\DELL\Desktop\Analytical-Decision-Modeling-2\519-case\dict_holi.csv', encoding='utf-8', index=False)

# The 2 lines below converts the dataframe to dictionary
#dict_clust_week = dict(zip(clust_week['Time_in_24'], labels_week))
#dict_clust_holi = dict(zip(clust_holi['Time_in_24'], labels_holi))

#print(dict_clust_week)
#print(dict_clust_holi)

# These lines below can be used to convert dictionary to dataframe again
#(pd.DataFrame.from_dict(data = dict_clust_week, orient='index').
#to_csv(r'C:\Users\DELL\Desktop\Analytical-Decision-Modeling-2\519-case\dict_week.csv', header = True))

#(pd.DataFrame.from_dict(data = dict_clust_holi, orient='index').
#to_csv(r'C:\Users\DELL\Desktop\Analytical-Decision-Modeling-2\519-case\dict_holi.csv', header = True))


# Now Plotting the scatter plots for the demand of electricity on weekends/holidays and weekdays
from matplotlib import pyplot as plt

plt.figure(figsize=(20, 10))
f2 = clust_week['Deseasonalized_demand'].values
f1 = clust_week['Time_in_24'].values
plt.scatter(f1, f2,s = 2, c='black')
plt.show()

plt.figure(figsize=(20, 10))
f4 = clust_holi['Deseasonalized_demand'].values
f3 = clust_holi['Time_in_24'].values
plt.scatter(f3, f4, s = 2, c='black')
plt.show()


# After analyzing the cluster labels from the csv output file we split the day into 6 clusters for weekdays and 5 clusters for weekend/holidays

# Printing calculated summary statistics for each of the cluster we made 
# based on these statistics we decide which distribution can be given for these clusters, normal, log, lognormal, uniform, triangular etc ..
print("Cluster 1")
Clust_1 = clust_week.loc[(clust_week['Time_in_24'] < 4)]
cluster1 = np.array(Clust_1['Deseasonalized_demand'])
mean1 = np.mean(cluster1, axis=0)
sd1 = np.std(cluster1, axis=0)
final_list_1 = [x for x in cluster1 if (x > mean1 - 2 * sd1)]
final_list_1 = [x for x in final_list_1 if (x < mean1 + 2 * sd1)]
print("min", min(final_list_1))
print("max", max(final_list_1))
print("mean", np.mean(final_list_1))
print("SD", np.std(final_list_1))
print("Range/SD", (max(final_list_1)-min(final_list_1))/np.std(final_list_1))
print()

print("Cluster 2")
Clust_2 = clust_week.loc[(clust_week['Time_in_24'] < 8) & (clust_week['Time_in_24'] > 3 )]
cluster2 = np.array(Clust_2['Deseasonalized_demand'])
mean2 = np.mean(cluster2, axis=0)
sd2 = np.std(cluster2, axis=0)
final_list_2 = [x for x in cluster2 if (x > mean2 - 2 * sd2)]
final_list_2 = [x for x in final_list_2 if (x < mean2 + 2 * sd2)]
print("min", min(final_list_2))
print("max", max(final_list_2))
print("mean", np.mean(final_list_2))
print("SD", np.std(final_list_2))
print("Range/SD", (max(final_list_2)-min(final_list_2))/np.std(final_list_2))
print()

print("Cluster 3")
Clust_3 = clust_week.loc[(clust_week['Time_in_24'] < 12) & (clust_week['Time_in_24'] > 7)]
cluster3 = np.array(Clust_3['Deseasonalized_demand'])
mean3 = np.mean(cluster3, axis=0)
sd3 = np.std(cluster3, axis=0)
final_list_3 = [x for x in cluster3 if (x > mean3 - 2 * sd3)]
final_list_3 = [x for x in final_list_3 if (x < mean3 + 2 * sd3)]
print("min", min(final_list_3))
print("max", max(final_list_3))
print("mean", np.mean(final_list_3))
print("SD", np.std(final_list_3))
print("Range/SD", (max(final_list_3)-min(final_list_3))/np.std(final_list_3))
print()

print("Cluster 4")
Clust_4 = clust_week.loc[(clust_week['Time_in_24'] < 16) & (clust_week['Time_in_24'] > 11)]
cluster4 = np.array(Clust_4['Deseasonalized_demand'])
mean4 = np.mean(cluster4, axis=0)
sd4 = np.std(cluster4, axis=0)
final_list_4 = [x for x in cluster4 if (x > mean4 - 2 * sd4)]
final_list_4 = [x for x in final_list_4 if (x < mean4 + 2 * sd4)]
print("min", min(final_list_4))
print("max", max(final_list_4))
print("mean", np.mean(final_list_4))
print("SD", np.std(final_list_4))
print("Range/SD", (max(final_list_4)-min(final_list_4))/np.std(final_list_4))
print()

print("Cluster 5")
Clust_5 = clust_week.loc[(clust_week['Time_in_24'] < 20) & (clust_week['Time_in_24'] > 15)]
cluster5 = np.array(Clust_5['Deseasonalized_demand'])
mean5 = np.mean(cluster5, axis=0)
sd5 = np.std(cluster5, axis=0)
final_list_5 = [x for x in cluster5 if (x > mean5 - 2 * sd5)]
final_list_5 = [x for x in final_list_5 if (x < mean5 + 2 * sd5)]
print("min", min(final_list_5))
print("max", max(final_list_5))
print("mean", np.mean(final_list_5))
print("SD", np.std(final_list_5))
print("Range/SD", (max(final_list_5)-min(final_list_5))/np.std(final_list_5))
print()

print("Cluster 6")
Clust_6 = clust_week.loc[(clust_week['Time_in_24'] < 24) & (clust_week['Time_in_24'] > 19)]
cluster6 = np.array(Clust_6['Deseasonalized_demand'])
mean6 = np.mean(cluster6, axis=0)
sd6 = np.std(cluster6, axis=0)
final_list_6 = [x for x in cluster6 if (x > mean6 - 2 * sd6)]
final_list_6 = [x for x in final_list_6 if (x < mean6 + 2 * sd6)]
print("min", min(final_list_6))
print("max", max(final_list_6))
print("mean", np.mean(final_list_6))
print("SD", np.std(final_list_6))
print("Range/SD", (max(final_list_6)-min(final_list_6))/np.std(final_list_6))
print()

print("Cluster 7")
Clust_7 = clust_holi.loc[(clust_holi['Time_in_24'] < 4)]
cluster7 = np.array(Clust_7['Deseasonalized_demand'])
mean7 = np.mean(cluster7, axis=0)
sd7 = np.std(cluster7, axis=0)
final_list_7 = [x for x in cluster7 if (x > mean7 - 2 * sd7)]
final_list_7 = [x for x in final_list_7 if (x < mean7 + 2 * sd7)]
print("min", min(final_list_7))
print("max", max(final_list_7))
print("mean", np.mean(final_list_7))
print("SD", np.std(final_list_7))
print("Range/SD", (max(final_list_7)-min(final_list_7))/np.std(final_list_7))
print()

print("Cluster 8")
Clust_8 = clust_holi.loc[(clust_holi['Time_in_24'] < 9) & (clust_holi['Time_in_24'] > 3)]
cluster8 = np.array(Clust_8['Deseasonalized_demand'])
mean8 = np.mean(cluster8, axis=0)
sd8 = np.std(cluster8, axis=0)
final_list_8 = [x for x in cluster8 if (x > mean8 - 2 * sd8)]
final_list_8 = [x for x in final_list_8 if (x < mean8 + 2 * sd8)]
print("min", min(final_list_8))
print("max", max(final_list_8))
print("mean", np.mean(final_list_8))
print("SD", np.std(final_list_8))
print("Range/SD", (max(final_list_8)-min(final_list_8))/np.std(final_list_8))
print()

print("Cluster 9")
Clust_9 = clust_holi.loc[(clust_holi['Time_in_24'] < 14) & (clust_holi['Time_in_24'] > 8)]
cluster9 = np.array(Clust_9['Deseasonalized_demand'])
mean9 = np.mean(cluster9, axis=0)
sd9 = np.std(cluster9, axis=0)
final_list_9 = [x for x in cluster9 if (x > mean9 - 2 * sd9)]
final_list_9 = [x for x in final_list_9 if (x < mean9 + 2 * sd9)]
print("min", min(final_list_9))
print("max", max(final_list_9))
print("mean", np.mean(final_list_9))
print("SD", np.std(final_list_9))
print("Range/SD", (max(final_list_9)-min(final_list_9))/np.std(final_list_9))
print()

print("Cluster 10")
Clust_10 = clust_holi.loc[(clust_holi['Time_in_24'] < 19) & (clust_holi['Time_in_24'] > 13)]
cluster10 = np.array(Clust_10['Deseasonalized_demand'])
mean10 = np.mean(cluster10, axis=0)
sd10 = np.std(cluster10, axis=0)
final_list_10 = [x for x in cluster10 if (x > mean10 - 2 * sd10)]
final_list_10 = [x for x in final_list_10 if (x < mean10 + 2 * sd10)]
print("min", min(final_list_10))
print("max", max(final_list_10))
print("mean", np.mean(final_list_10))
print("SD", np.std(final_list_10))
print("Range/SD", (max(final_list_10)-min(final_list_10))/np.std(final_list_10))
print()

print("Cluster 11")
Clust_11 = clust_holi.loc[(clust_holi['Time_in_24'] < 24) & (clust_holi['Time_in_24'] > 18)]
cluster11 = np.array(Clust_11['Deseasonalized_demand'])
mean11 = np.mean(cluster11, axis=0)
sd11 = np.std(cluster11, axis=0)
final_list_11 = [x for x in cluster11 if (x > mean11 - 2 * sd11)]
final_list_11 = [x for x in final_list_11 if (x < mean11 + 2 * sd11)]
print("min", min(final_list_11))
print("max", max(final_list_11))
print("mean", np.mean(final_list_11))
print("SD", np.std(final_list_11))
print("Range/SD", (max(final_list_11)-min(final_list_11))/np.std(final_list_11))
print()
