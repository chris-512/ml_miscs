import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools import plotting
from scipy import stats
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
from scipy import stats

import os
print(os.listdir('../input'))

# read data as pandas data frame
data = pd.read_csv('../input/data.csv')
data = data.drop(['Unnamed: 32', 'id'], axis=1)

# quick look to data
print(data.head())
print(data.shape) # (569, 31)
print(data.columns)

"""Histogram

 - How many times each value appears in dataset. This description is called the distribution of variable
 - Most common way to represent distribution of variable is histogram that is graph which shows frequency of 
   each value.
 - Frequency = number of times each value appears
 - Example: [1, 1, 1, 1, 2, 2, 2]. Frequency of 1 is four and frequency of 2 is three.
"""
m = plt.hist(data[data['diagnosis'] == 'M'].radius_mean, bins=30, fc=(1, 0, 0, 0.5), label='Malignant')
b = plt.hist(data[data['diagnosis'] == 'B'].radius_mean, bins=30, fc=(0, 1, 0, 0.5), label='Benign')
plt.legend()
plt.xlabel('Radius Mean Values')
plt.ylabel('Frequency')
plt.title('Histogram of Radius Mean for Benign and Malignant Tumors')
plt.show()

frequent_malignant_radius_mean = m[0].max
index_frequent_malignant_radius_mean = list(m[0]).index(frequent_malignant_radius_mean)
most_frequent_malignant_radius_mean = m[1][index_frequent_malignant_radius_mean]
print('Most frequent malignant radius mean is: ', most_frequent_malignant_radius_mean)

"""Outliers

  - While looking histogram as you can see there are rare values in benign distribution (green in graph)
  - There values can be errors or rare events.
  - These errors and rare events can be called outliers.
  - Calculating outliers:
    - first we need to calculate first quartile (Q1) (25%)
    - then find IQR (inter quartile range) = Q3-Q1
    - finally compute Q1 - 1.5IQR and Q3 + 1.5IQR
    - Anything outside this range is an outlier
    - lets write the code for benign tumor distribution for feature radius mean
"""
data_benign = data[data['diagnosis'] == 'B']
data_malignant = data[data['diagnosis'] == 'M']
desc = data_benign.radius_mean.describe()
Q1 = desc[4]
Q3 = desc[6]
IQR = Q3-Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
print('Anything outside this range is an outlier: (', lower_bound, ', ', upper_bound, ')')
data_benign[data_benign.radius_mean < lower_bound].radius_mean
print('Outliers: ', data_benign[(data_benign.radius_mean < lower_bound) | (data_benign.radius_mean > upper_bound)].radius_mean.values)

"""BOX PLOT
 - You can see outliers also from box plots
 - We found 3 outlier in benign radius mean and in box plot there are 3 outliers.
"""
melted_data = pd.melt(data, id_vars='diagnosis', value_vars=['radius_mean', 'texture_mean'])
plt.figure(figsize=(15, 10))
sns.boxplot(x='variable', y='value', hue='diagnosis', data=melted_data)
plt.show()

"""SUMMARY STATISTICS
 - Mean
 - Variance: spread of distribution
 - Standard deviation square root of variance
 - Lets look at summary statistics of benign tumor radiance mean
"""
print('mean: ', data_benign.radius_mean.mean())
print('variance: ', data_benign.radius_mean.var())
print('standard deviation (std): ', data_benign.radius_mean.std())
print('describe method: ', data_benign.radius_mean.describe())

"""CDF
 - Cumulative distribution function is the probability that the variable takes a value less than or equal to x. P(X <= x)
 - Lets explain in cdf graph of benign radiuses mean
 - in graph, what is P(12 < X)? The answer is 0.5. 
 - You can plot cdf with two different method
"""
plt.hist(data_benign.radius_mean, bins=50, fc=(0, 1, 0, 0.5), label='Benign', normed=True, cumulative=True)
sorted_data = np.sort(data_benign.radius_mean)
y = np.arange(len(sorted_data)/float(len(sorted_data)-1))
plt.plot(sorted_data, y, color='red')
plt.title('CDF of being tumor radius mean')
plt.show()

"""Effect size
 - One of the summary statistics.
 - It describes size of an effect. It is simple way of quantifying the difference between two groups.
 - In an other saying, effect size emphasizes the size of the difference.
 - Use cohen effect size
 - Cohen suggest that if d(effect_size) = 0.2, it is small effect size, d = 0.5 medium effect size, d = 0.8 large effect size.
 - lets compare size of the effect between benign radius mean and malignant radius mean
 - Effect size is 2.2 that is too big and says that two groups are different from each other as we expect. Because our groups
  are benign radius mean and malignant radius mean that are different from each other.
"""
mean_diff = data_malignant.radius_mean.mean() - data_benign.radius_mean.mean()
var_benign = data_benign.radius_mean.var()
var_malignant = data_malignant.radius_mean.var()
var_pooled = (len(data_benign) * var_benign + len(data_malignant) * var_malignant) / float(len(data_benign) + len(data_malignant))
effect_size = mean_diff / np.sqrt(var_pooled)
print('Effect size: ', effect_size)

"""Relationship Between Variables

 - We can say that two variables are related with each other, if one of them gives information about others
 - For example, price and distance. If you go long distance with taxi you will pay more. Therefore  we can say
   that price and distance are positively related with each other.
 - Scatter Plot
 - Simplest way to check relationship between two variables.
 - Lets look at relationship between radius mean and area mean
 - In scatter plot you can see that when radius mean increases, area mean also increases. Therefore, they are
   positively correlated with each other.
 - There is no correlation between area mean and fractal dimension se. Because when area mean changes,
  fractal dimension se is not affected by chance of area mean.

"""
plt.figure(figsize=(15, 10))
sns.jointplot(data.radius_mean, data.area_mean, kind='regg')
plt.show()

# Also we can look relationship between more than 2 distributions
sns.set(style = 'white')
df = data.loc[:, ['radius_mean', 'area_mean', 'fractal_dimension_se']]
g = sns.PairGrid(df, diag_sharey=False,)
g.map_lower(sns.kdeplot, cmap='Blues_d')
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)
plt.show()

"""CORRELATION

"""