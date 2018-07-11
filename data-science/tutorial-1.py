import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # visualization

from subprocess import check_output
print(check_output(["ls", "../input"]).decode('utf-8'))

data = pd.read_csv("../input/pokemon.csv")
print(data.info())
print(data.corr())
# correlation map
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()

print(data.head())
print(data.columns)

'''MATPLOTLIB
Matplotlib is a python library that helps us to plot data.
The easiest and basic plots are line, scatter and histogram plots.

 - Line plot is better than when x axis is time.
 - Scatter is better when there is correlation between two variables.
 - Histogram is better when we need to see distributions of numerical data.
 - Customization: Colors, label, thickness of line, title, opacity, grid, figsize, ticks of axis and linestyle
'''
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestype = linestyle of line
data.Speed.plot(kind = 'line', color = 'g', label = 'Speed', linewidth=1, alpha=0.5, grid=True, linestyle=':')
data.Defense.plot(color='r', label='Defense', linewidth=1, alpha=0.5, grid=True, linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()

# Scatter Plot
# x = attack, y = defense
data.plot(kind='scatter', x='Attack', y='Defense', alpha=0.5, color='red')
plt.xlabel('Attack')  # lable = name of label
plt.ylabel('Defense')
plt.title('Attack Defense Scatter Plot') # title = title of plot
plt.show()

# Histogram
# bins = number of bar in figure
data.Speed.plot(kind='hist', bins=50, figsize=(12, 12))
plt.show()

# clf() = cleans it up again you can start a fresh
data.Speed.plot(kind='hist', bins=50)
plt.clf()
# We cannot see plot due to clf()


"""Pandas

What we need to know about pandas?
 - CSV: comma - separated values
"""
data = pd.read_csv("../input/pokemon.csv")
series = data['Defense'] # data['Defense'] = series
print(type(series))
data_frame = data[['Defense']] # data[['Defense']] = data frame
print(type(data_frame))

# 1 - Filtering Pandas data frame
x = data['Defense'] > 200 # There are only 3 pokemons who have higher defense value than 200
print(data[x])

# 2 - Filtering pandas with logical_and
# There are only 2 pokemons who have higher defense value than 200 and higher attack value than 100
x = data[np.logical_and(data['Defense']>200, data['Attack']>100)]
print(data[x])

# This is also same with previous code line. Therefore we can also use '&' for filtering.
print(data[(data['Defense']>200) & (data['Attack']>100)])

# For pandas we can achieve index and value
for index, value in data[['Attack']][0:1].iterrows():
    print(index, " : ", value)

"""USER DEFINED FUNCTION
"""
# example of what we learn above
def tuple_ex():
    """ return defined t tuple """
    t = (1, 2, 3)
    return t
a, b, c = tuple_ex()
print(a, b, c)

"""SCOPE
"""
# guess print what
x = 2
def f():
    x = 3
    return x
print(x) # x = 2 global scope
print(f()) # x = 3 local scope

# What if there is no local scope
x = 5
def f():
    y = 2*x # there is no local scope x
    return y
print(f()) # it uses global scope x
# First local scope searched, then global scope searched, if two of them cannot be found
# built-in scope searched

# How can we learn what is built-in scope
import builtins
print(dir(builtins))

"""NESTED FUNCTION
 - function inside function
 - There is a LEGB rule that is search local scope, enclosing function, global and built in scopes, respectively.
"""
def square():
    """return square of value"""
    def add():
        """add two local variable"""
        x = 2
        y = 3
        z = x + y
        return z
    return add() ** 2
print(square())

"""DEFAULT and FLEXIBLE ARGUMENTS
 - Default argument example
  def f(a, b=1): # b = 1 is default argument

 - Flexible argument example
  def f(*args): # *args can be one or more
  
  def f(**kwargs): # **kwargs is a dictionary
  
"""

# default arguments
def f(a, b=1, c=2):
    y = a + b + c
    return y
print(f(5))
# what if we want to change default arguments
print(f(5, 4, 3))

# flexible arguments *args
def f(*args):
    for i in args:
        print(i)

print(f(1))
print("")
print(f(1, 2, 3, 4))
# flexible arguments **kwargs that is dictionary
def f(**kwargs):
    """print key and value of dictinary"""
    for key, value in kwargs.items():
        print(key, " ", value)

print(f(country='spain', capital='madrid', population=123456))

# Lambda function
square = lambda x: x ** 2 # where x is name of argument
print(square(4))
tot = lambda x, y, z: x + y + z # where x, y, z are names of arguments
print(tot(1, 2, 3))

# Anonymous function
number_list = [1, 2, 3]
y = map(lambda x: x ** 2, number_list)
print(list(y))

# Iterators
# iterable: an object with an associated iter() method ex) list, strings, and dictionaries
name = 'ronaldo'
it = iter(name)
print(next(it)) # print next iteration
print(*it) # print remaining iteration

list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]
z = zip(list1, list2)
print(z)
z_list = list(z)
print(z_list)

un_zip = zip(*z_list)
un_list1, un_list2 = list(un_zip)
print(un_list1)
print(un_list2)
print(type(un_list2))

# List Comprehension
# list comprehension: collapse for loops for building lists into a single line
# Example of list comprehension
num1 = [1, 2, 3]
num2 = [i + 1 for i in num1]
print(num2)

# Conditionals on iterable
num1 = [5, 10, 15]
num2 = [i ** 2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

# lets return pokemon csv and make one more list comprehension example
# lets classify pokemons whether they have high or low speed. Our threshold is average speed.
threshold = sum(data.Speed) / len(data.Speed)
data['speed_level'] = ['high' if i > threshold else 'low' for i in data.Speed]
print(data.loc[:10, ['speed_level', 'Speed']]) # we will learn loc more detailed later


"""CLEANING DATA
 DIAGNOSE DATA for CLEANING

 we need to diagnose and clean data before exploring
 Unclean data:
  - Column name inconsistency like upper-lower case letter or space between words
  - missing data
  - different language
"""

data = pd.read_csv('../input/pokemon.csv')
print(data.head()) # head shows first 5 rows
print(data.tail()) # tail shows last 5 rows
print(data.columns) # columns gives column name of features
print(data.shape) # shape gives number of rows and columns in a table
# info gives data type like dataframe, number of sample or row, number of feature or column,
# feature types and memory usage
print(data.info())

"""EXPLORATORY DATA ANALYSIS

value_counts(): Frequency counts
outliers: the value that is considerably higher or lower from rest of the data

 - Lets say value at 75% is Q3 and value at 25% is Q1.
 - Outliers are smaller than Q1 - 1.5 (Q3-Q1) and bigger than Q3 + 1.5 (Q3-Q1). (Q3-Q1)=IQR
   We will use describe() method. Describe method includes:
    - count: number of entries
    - mean: average of entries
    - std: standard deviation
    - min: minimum entry
    - 25%: first quantile
    - 50%: median or second quantile
    - 75%: third quantile
    - max: maximum entry

 What is quantile?
  - 1, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17
  - The median is the number that is in middle of the sequence. In this case, it would be 11.
  - The lower quartile is the median in between the smallest number and the median i.e. in between 1 and 11, which is 6.
  - The upper quartile, you find the median between the median and the largest number i.e. between 11 and 17, which will be 14 according to the question above. 

"""
# For example lets took frequency of pokemon types
print(data['Type 1'].value_counts(dropna=False)) # if there are nan values that also be counted
# As it can be seen below there are 112 water pokemon or 70 grass pokemon

# For example max HP is 255 or min defense is 5
print(data.describe()) # ignore null entries

"""VISUAL EXPLORATORY DATA ANALYSIS
 - Box plots: visualize basic statistics like outliers, min/max or quantiles
"""

# For example: compare attack of pokemons that are legendary or not
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='Attack', by='Legendary')

"""TIDY DATA

  We tidy data with melt(). Describing melt is confusing. Therefore lets make example to understand it.
  
"""
# Firstly I create new data from pokemons data to explain melt more easily.
data_new = data.head() # I only take 5 rows into new data
print(data_new)

# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new, id_vars='Name', value_vars=['Attack', 'Defense'])
print(melted)

"""PIVOTING DATA

 Reverse of melting
"""
# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index='Name', columns='variable', values='value')

"""CONCATENATING DATA

 We can concatenate two dataframes
"""
# Firstly lets create 2 data frame
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1, data2], axis=0, ignore_index=True) # axis = 0 : adds dataframes in row
print(conc_data_row)

data1 = data['Attack'].head()
data2 = data['Defense'].head()
conc_data_col = pd.concat([data1, data2], axis=1) # axis = 1: adds dataframes in col
print(conc_data_col)

"""DATA TYPES
 
 There are 5 basic data types: object(string), booleab, integer, float and categorical.
 We can make conversion data types like from str to categorical or from int to float
 Why is category important:
  - make dataframe smaller in memory
  - can be utilized for analysis especially for sklear (we will learn later)
"""
print(data.dtypes)

# lets convert object(str) to categorical and int to float.
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')

# As you can see Type 1 is converted from object to categorical
# And Speed is converted from int to float
print(data.dtypes)

"""MISSING DATA and TESTING WITH ASSERT

"""




"""PANDAS FOUNDATION

 - single column = series 
 - NaN = not a number
 - dataframe.values = numpy
"""

# data frames from dictionary
country = ['Spain', 'France']
population = ['11', '12']
list_label = ['country', 'population']
list_col = [country, population]
zipped = list(zip(list_label, list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
print(df)

# Add new columns
df['capital'] = ['madrid', 'paris']
print(df)

# Broadcasting
df['income'] = 0 # Broadcasting entire columns
print(df)

"""VISUAL EXPLORATORY DATA ANALYSIS
 - Plot
 - Subplot
 - Histogram:
   - bins: number of bins
   - range(tuple): min and max values of bins
   - normed(boolean): normalize or not
   - cumulative(boolean): compute cumulative distribution 
"""
# Plotting all data
data1 = data.loc[:, ['Attack', 'Defense', 'Speed']]
# it is confusing
data1.plot()

# subplots
data1.plot(subplots=True)
plt.show()

# scatter plot
data1.plot(kind='scatter', x='Attack', y='Speed')
plt.show()

# hist plot
data1.plot(kind='hist', y='Defense', bins=50, range=(0, 250), normed=True)

# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2, ncols=1)
data1.plot(kind='hist', y='Defense', bins=50, range=(0, 250), normed=True, ax=axes[0])
data1.plot(kind='hist', y='Defense', bins=50, range=(0, 250), normed=True, ax=axes[1], cumulative=True)
plt.savefig('graph.png')
print(plt)

"""STATISTICAL EXPLORATORY DATA ANALYSIS

I already explained it at previous parts. However lets look at one more time.
 - count: number of entries
 - mean: average of entries
 - std: standard deviation
 - min: minimum entry
 - 25%: first quantile
 - 50%: median or second quantile
 - 75%: third quantile
 - max: maximum entry
"""
data.describe()

"""INDEXING PANDAS TIME SERIES
 - datatime = object
 - parse_dates(boolean): Transform data to ISO 8601 (yyyy-mm-dd hh:mm:ss) format
"""
time_list = ['1992-03-08', '1992-04-12']
print(type(time_list[1]))
# however we want it to be datatime object
datetime_object = pd.to_datatime(time_list)
print(type(datetime_object))

# close warnings
import warnings
warnings.filterwarnings('ignore')

# In order to practice lets take head of pokemon data and add it a time list
data2 = data.head()
data_list = ['1992-01-10', '1992-02-10', '1993-03-15', '1993-03-16']
datatime_object = pd.to_datetime(date_list)
data2['date'] = datatime_object
# lets make date as index
data2 = data2.set_index('date')
print(data2)

# Now we can select according to our date index
print(data2.loc['1993-03-16'])
print(data2.loc['1992-03-10', '1993-03-16'])

"""RESAMPLING PANDAS TIME SERIES

 - Resampling: statistical method over different time intervals
  - Needs string to specify frequency like 'M'=month or 'A'=year
 - Downsampling: reduce date time rows to slower frequency like from daily to weekly
 - Upsampling: increase data time rows to faster frequency like from daily to hourly
 - Interpolate: Interpolate values according to different methods like 'linear', 'time', or 'index'
   - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.interpolate.html
"""
# We will use data2 that we create at previous part
data2.resample('A').mean()

# Lets resample with month
data2.resample('M').mean()
# As you can see there are a lot of nan because data2 does not include all months

# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate
# We can interpolate from first value
data2.resample('M').first().interpolat('linear')

# Or we can interpolate with mean()
data2.resample('M').mean().interpolate('linear')

"""MANIPULATING DATA FRAMES WITH PANDAS

 INDEXING DATA FRAMES
  - Indexing using square brackets
  - Using column attribute and row label
  - Using loc accessor
  - Selecting only some columns 
"""
# read data
data = pd.read_csv('../input/pokemon.csv')
data = data.set_index('#')
print(data.head())

# indexing using square brackets
print(data['HP'][1])

# using column attribute and row label
print(data.HP[1])

# using loc accessor
print(data.loc[1, ['HP']])

# Selecting only some columns
data[['HP', 'Attack']]


'''Slicing Data Frame

 - Difference between selecting columns
  - Series and data frames
 - Slicing and indexing series
 - Reverse slicing
 - From something to end
 
'''
# Difference between selecting columns: series and dataframes
print(type(data['HP'])) # series
print(type(data[['HP']])) # data frames

# Slicing and indexing series
data.loc[1:10, 'HP':'Defense'] # 10 and 'Defense' are inclusive

# Reverse slicing
data.loc[10:1:-1, "HP":"Defense"]

# From something to end
data.loc[1:10, "Speed":]

"""FILTERING DATA FRAMES
 Creating boolean series Combining filters Filtering column based others 
"""
# Creating boolean series
boolean = data.HP > 200
print(data[boolean])

# Combining filters
first_filter = data.HP > 150
second_filter = data.Speed > 35
print(data[first_filter & second_filter])

# Filtering column based others
print(data.HP[data.Speed < 15])

"""Transforming data
 - Plain python functions
 - Lambda function: to apply arbitrary python function to every element
 - Defining column using other columns
"""
# Plain python functions
def div(n):
    return n/2

data.HP.appy(div)

# Or we can use lambda function
data.HP.apply(lambda n : n /2)

# Defining column using other columns
data['total_power'] = data.Attack + data.Defense
print(data.head())

"""INDEXING OBJECTS AND LABELED DATA
"""
# our index name is this:
print(data.index.name)
# lets change it
data.index_name = 'index_name'
print(data.head())

# Overwrite index
# if we want to modify index we need to change all of them
data.head()
# first copy of our data to data3 then change index
data3 = data.copy()
# lets make index start from 100. It is not remarkable change but it is just simple.
data3.index = range(100, 900, 1) # index changed
print(data3.head())

'''
We can make one of the column as index. I actually did it at the beginning of manipulating data frames
with pandas section

It was like this

 data = data.set_index('#')
also you can use
 data.index = data['#']

'''
# HIERARCHICAL INDEXING
# lets read data frame one more time to start from beginning
data = pd.read_csv('../input/pokemon.csv')
print(data.head())

# Setting index: type 1 is outer / type 2 is inner index
data1 = data.set_index(['Type 1', 'Type 2'])
data1.head(100)
# data1.loc['Fire', 'Flying'] # how to use indexes

"""PIVOTING DATA FRAMES

 - pivoting: reshape tool
"""
# key = column, value = list of row data
dic = {'treatment': ['A', 'A', 'B', 'B'], 'gender': ['F', 'M', 'F', 'M'], 'response': [10, 45, 5, 9], 'age': [15, 4, 72, 65]}
df = pd.DataFrame(dic)
print(df)

# pivoting
# index = row, columns = column
df.pivot(index='treatment', columns='gender', values='response')

"""STACKING and UNSTACKING DATAFRAME

 - deal with multi label indexes 
 - level: position of unstacked index
 - swaplevel: change inner and outer level index position
"""
df1 = df.set_index(['treatment', 'gender'])
print(df1)

# lets unstack it
# level determines indexes
df1.unstack(level=0)

df1.unstack(level=1)

# change inner and outer level index position
df2 = df1.swaplevel(0, 1)
print(df2)

"""MELTING DATA FRAMES
 - reverse of pivoting
"""
print(df)
pd.melt(df, id_vars='treatment', value_vars=['age', 'response'])

"""CATEGORICAL AND GROUPBY
"""
# We will use df
print(df)
# according to treatment, take means of other features
df.groupby('treatment').mean() # mean is aggregation / reduction method
# there are other methods like sum, std, max, or min

# we can only choose one of the feature
df.groupby('treatment').age.max()

# or we can choose multiple features
df.groupby('treatment')[['age', 'response']].min()
df.info()
# As you can see gender is object
# However if we use groupby, we can convert it categorial data
# Because categorical data uses less memory, speed up operations like groupby
# df['gender'] = df['gender'].astype('category')
# df['treatment'] = df['treatment'].astype('category')
# df.info()














