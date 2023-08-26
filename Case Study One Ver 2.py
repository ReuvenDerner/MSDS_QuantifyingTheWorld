#!/usr/bin/env python
# coding: utf-8

# __Title:__ Case Study 1: Linear Regression  
# __Authors:__ Will Butler, Robert (Reuven) Derner 
# __Date:__ 8/24/23 

# ## Business Understanding
# 
# We have a problem that has been brought to us from a group of scientists that are looking at superconductors. Superconductors are materials that give little or no resistance to electrical current. 
# 
# The Scientists are looking at us to use the data provided to produce a model to predict new superconductors based on the properties and the data that they have found so far. Some of the data points include material composition, temperature at which they superconduct. We're going to examine the data set through exploratory data analysis. 
# 
# The model desired is going to predict new superconductors and the temperature at which they operate based on the experimental inputs from the data that they have provided to us. The model needs to be interpretable so that the scientists' can figure out at what temperature new superconductors would become superconductors, not only if they would be superconductors. We will conduct a regression type of model to give the scientists ease of interpretability based on the relative importance of each feature in the model. 
#  
# 
# Data Source:
# 
# Provided by client with metadata dictionary regarding terms 

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
import random
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# Workbook settings
pd.set_option('display.max_columns', None)
random.seed(110)
pio.renderers.default='notebook'


# In[ ]:


# Import data from github (next phase)
url = 'https://github.com/cdholmes11/MSDS-7331-ML1-Labs/blob/main/Lab-1_Visualization_DataPreprocessing/data/Combined_Flights_2021_sample.csv?raw=true'
flight_data_df = pd.read_csv(url, encoding = "utf-8")


# In[2]:


# loacal Import (To be removed later)
data = pd.read_csv("C:/Users/robert.derner/OneDrive - Flagship Credit Acceptance/Documents/School/Quantifying The World/Case Study One/train.csv")


# ## Data Meaning Type
# 
# Describe the meaning and type of data (scale, values, etc.) for each attribute in the data file.

# | Response Features | Definition                                                                                                 |
# | ----------------- | ---------------------------------------------------------------------------------------------------------- |
# | Critical Temp         | When a superconductor reaches critical temperature and becomes a superconductor                                                                         |

# | Categorical Features | Definition                                                                                                 |
# | ----------------- | ---------------------------------------------------------------------------------------------------------- |
# | Number_of_elements         | The number of periodic elements contained in the superconductor                                                                         |

# | Continuous Features | Definition                                                                                                 |
# | ------------------- | ---------------------------------------------------------------------------------------------------------- |
# | mean_atomic_mass   | The average atomic mass                                                                          |
# wtd_mean_atomic_mass             | The weighted average atomic mass                                                                                    |
# gmean_atomic_mass             | g-average given the atomic mass                                                                                    |
# wtd_gmean_atomic_mass             | Weighted g-average given the atomic mass                                                                                   |
# entropy_atomic_mass             | The degree of disorder or uncertainy given the atomic mass                                                                                    |
# wtd_entropy_atomic_mass             | Weighted average degree of disorder or uncertainty given the atomic mass                                                                                    |
# range_atomic_mass             | Range of atomic mass                                                                                   |
# wtd_range_atomic_mass             | Weight Range of atomic mass                                                                                    |
# std_atomic_mass             | Standard Deviation of atomic mass                                                                                    |
# wtd_std_atomic_mass             | Weighted standard deviation of atomic mass                                                                                    |
# mean_fie             | Average of Fie                                                                                    |
# wtd_mean_fie             | Weighted average of fie                                                                                    |
# gmean_fie             | G-Average of Fie                                                                                    |
# wtd_gmean_fie             | Weighted g-average of fie                                                                                    |
# entropy_fie             | The degree of disorder or uncertainty  of Fie                                                                                    |
# wtd_entropy_fie             | Weighted degree of disorder or uncertainty of Fie                                                                                    |
# range_fie             | Range of FIE                                                                                    |
# wtd_range_fie             | Weighted Range of FIE                                                                                    |
# std_fie             | Standard deviation of FIE                                                                                    |
# wtd_std_fie             | Weighted Standard Deviation of FIE                                                                                    |
# mean_atomic_radius             | The Average of atomic radius                                                                                    |
# wtd_mean_atomic_radius             | The weighted average of atomic radius                                                                                    |
# gmean_atomic_radius             | The g-average of atomic radius                                                                                    |
# wtd_gmean_atomic_radius             | The weighted g-average of atomic radius                                                                                    |
# entropy_atomic_radius             | The degree of disorder or uncertainty of atomic radius                                                                                    |
# wtd_entropy_atomic_radius             | The weighted degree of disorder or uncertainty of atomic radius                                                                                    |
# range_atomic_radius             | The range of atomic radius                                                                                    |
# wtd_range_atomic_radius             | The weighted range of atomic radius                                                                                    |
# std_atomic_radius             | The standard deviation of atomic radius                                                                                    |
# wtd_std_atomic_radius             | The weighted standard deviation of atomic radius                                                                                   |
# mean_Density             | The average Density                                                                                    |
# wtd_mean_Density             | The weighted average Density                                                                                    |
# gmean_Density             | The g-average Density                                                                                    |
# wtd_gmean_Density             | The weghted g-average Density                                                                                    |
# entropy_Density             | The degree of disorder or uncersity in Density                                                                                    |
# wtd_entropy_Density             | The weighted degree of disorder or uncertainty in Density                                                                                    |
# range_Density             | The range of Density                                                                                   |
# wtd_range_Density             | The weighted range of Density                                                                                    |
# std_Density             | The standard deviation of Density                                                                                    |
# wtd_std_Density             | The weighted standard deviation of Density                                                                                    |
# mean_ElectronAffinity             | The average of Electron Affinity                                                                                    |
# wtd_mean_ElectronAffinity             | The weighted average of Electron Affinity                                                                                   |
# entropy_ElectronAffinity             | The degree of disorder or uncersity in Electron Affinity                                                                                    |
# wtd_entropy_ElectronAffinity             | The weighted degree of disorder or uncertainty in Electron Affinity                                                                                    |
# range_ElectronAffinity             | The range of Electron Affinity                                                                                    |
# wtd_range_ElectronAffinity             | The weighted range of Electron Affinity                                                                                    |
# std_ElectronAffinity             | The standard deviation of Electron Affinity                                                                                    |
# wtd_std_ElectronAffinity             | The wegithed standard deviation of Electron Affinity                                                                                    |
# mean_FusionHeat             | The average of Fusion Heat                                                                                    |
# wtd_mean_FusionHeat             | The weighted average of Fusion Heat                                                                                    |
# gmean_FusionHeat             | The g-average of of Fusion Heat                                                                                    |
# wtd_gmean_FusionHeat             | The weighted g-average of Fusion Heat                                                                                    |
# entropy_FusionHeat             | The degree o fdisorder or uncertainty of Fusion Heat                                                                                    |
# wtd_entropy_FusionHeat             | The weighted degree of disorder or uncertainity of Fusion Heat                                                                                    |
# range_FusionHeat             | Flight Time, in Minutes                                                                                    |
# wtd_range_FusionHeat             | Flight Time, in Minutes                                                                                    |
# std_FusionHeat             | Flight Time, in Minutes                                                                                    |
# wtd_std_FusionHeat             | Flight Time, in Minutes                                                                                    |
# mean_ThermalConductivity             | Flight Time, in Minutes                                                                                    |
# wtd_mean_ThermalConductivity             | Flight Time, in Minutes                                                                                    |
# gmean_ThermalConductivity             | Flight Time, in Minutes                                                                                    |
# wtd_gmean_ThermalConductivity             | Flight Time, in Minutes                                                                                    |
# entropy_ThermalConductivity             | Flight Time, in Minutes                                                                                    |
# wtd_entropy_ThermalConductivity             | Flight Time, in Minutes                                                                                    |
# range_ThermalConductivity             | Flight Time, in Minutes                                                                                    |
# wtd_range_ThermalConductivity             | Flight Time, in Minutes                                                                                    |
# std_ThermalConductivity             | Flight Time, in Minutes                                                                                    |
# wtd_std_ThermalConductivity             | Flight Time, in Minutes                                                                                    |
# mean_Valence             | Flight Time, in Minutes                                                                                    |
# wtd_mean_Valence             | Flight Time, in Minutes                                                                                    |
# gmean_Valence             | Flight Time, in Minutes                                                                                    |
# wtd_gmean_Valence             | Flight Time, in Minutes                                                                                    |
# entropy_Valence             | Flight Time, in Minutes                                                                                    |
# wtd_entropy_Valence             | Flight Time, in Minutes                                                                                    |
# range_Valence             | Flight Time, in Minutes                                                                                    |
# wtd_range_Valence             | Flight Time, in Minutes                                                                                    |
# std_Valence             | Flight Time, in Minutes                                                                                    |
# wtd_std_Valence             | Flight Time, in Minutes                                                                                    |
#    |

# 

# 

# ## Data Quality
# Verify data quality: Explain any missing values, duplicate data, and outliers. Are those mistakes? How do you deal with these problems? Give justifications for your methods.

# In[4]:


data.shape


# __Missing Values__  
# The dataset contains no missing values. There is nothing for us to imputate or reshape. 
# 
# For the purposes of regression on the Critical Temp, we may proceed with the analysis.

# In[7]:


# Features with Null Values and Percent missing
null_df = pd.DataFrame(data[data.columns[data.isnull().any()]].isnull().sum()).reset_index()
null_df.columns = ['Feature', 'Value']
null_df['Percent'] = round((null_df['Value'] / data.shape[0] * 100),2)

null_df


# __Duplicate Values__  
# There are 66 duplicate values in the data set. No action was needed.

# In[8]:


# Duplicate record validation
data.duplicated().sum()


# __Data Type Conversion__  
# In this section we grouped all features by their correct data type and converted each to their coresponding group. This facilitates a much easier analysis into the statistics of each feature type.

# In[8]:


# Features grouped by data type
cat_features = ['number_of_elements']
cont_features = ['mean_atomic_mass','wtd_mean_atomic_mass','gmean_atomic_mass','wtd_gmean_atomic_mass',
                 'entropy_atomic_mass','wtd_entropy_atomic_mass','range_atomic_mass','wtd_range_atomic_mass','std_atomic_mass',
                 'wtd_std_atomic_mass','mean_fie', 'wtd_mean_fie','gmean_fie','wtd_gmean_fie','entropy_fie','wtd_entropy_fie',
                 'range_fie','wtd_range_fie','std_fie','wtd_std_fie','mean_atomic_radius','wtd_mean_atomic_radius',
                 'gmean_atomic_radius','wtd_gmean_atomic_radius','entropy_atomic_radius','wtd_entropy_atomic_radius',
                 'range_atomic_radius','wtd_range_atomic_radius','std_atomic_radius','wtd_std_atomic_radius','mean_Density',
                 'wtd_mean_Density','gmean_Density','wtd_gmean_Density','entropy_Density','wtd_entropy_Density','range_Density',
                 'wtd_range_Density','std_Density','wtd_std_Density','mean_ElectronAffinity','wtd_mean_ElectronAffinity',
                 'gmean_ElectronAffinity','wtd_gmean_ElectronAffinity','entropy_ElectronAffinity','wtd_entropy_ElectronAffinity',
                 'range_ElectronAffinity','wtd_range_ElectronAffinity','std_ElectronAffinity','wtd_std_ElectronAffinity',
                 'mean_FusionHeat','wtd_mean_FusionHeat','gmean_FusionHeat','wtd_gmean_FusionHeat','entropy_FusionHeat',
                 'wtd_entropy_FusionHeat','range_FusionHeat','wtd_range_FusionHeat','std_FusionHeat','wtd_std_FusionHeat',
                 'mean_ThermalConductivity','wtd_mean_ThermalConductivity','gmean_ThermalConductivity',
                 'wtd_gmean_ThermalConductivity','entropy_ThermalConductivity','wtd_entropy_ThermalConductivity',
                 'range_ThermalConductivity','wtd_range_ThermalConductivity','std_ThermalConductivity',
                 'wtd_std_ThermalConductivity','mean_Valence','wtd_mean_Valence','gmean_Valence','wtd_gmean_Valence',
                 'entropy_Valence','wtd_entropy_Valence','range_Valence','wtd_range_Valence','std_Valence','wtd_std_Valence']


# In[9]:


# Features converted to corresponding group type (may be interesting to convert to string)
# data[cat_features] = data[cat_features].astype("string")


# __Outliers__  
# Several of our continuous variables appear to have outliers. We'll explore this later in more detail in the Simple Statistics section. In this section, our primary focus will be given to the dependent feature, DepDelayMinutes. It's heavily right skewed and appears to contain many outliers. Once we review the data more deeply, we'll be able to make a better assessment on whether to drop the outliers or attempt to transform them. For perspective, a day consists of 1440 minutes. In our research, there does not appear to be an agreed upon standard for how long a flight can be delayed before classifying it as cancelled.
# 
# Since flights that were not delayed will not be included in the regression portion of our model, we've filtered the data set to only include delayed flights. As you can see below, even when removing the 0-minute observations from non-delayed flights, the distribution is heavily right skewed and contains many outliers. When we get to the New Feature section of our analysis, we will introduce several transformations for this feature and reevaluate both the skewness and the outliers. Given the lack of standardization from the business and the lack of information on the influence, we've decided to wait till the regression portion of our analysis to make any firm decisions on how to handle these outliers.

# In[10]:


# Box Plot - Critical Temp by Number of Elements (string)
fig = px.box(data[data['critical_temp']==True], x='number_of_elements',
             width=800, height=400, title='Box Plot -Number of Elements')
fig.show()


# The above table shows only 11 observations delayed more than a day. A more appropriate method might be to flag which observations were delayed into the following day. This might provide a more appropriate cap to this feature that is more in line with the upper quartile of 96 minutes.

# ## Simple Statistics
# Visualize appropriate statistics (e.g., range, mode, mean, median, variance, counts) for a subset of attributes. Describe anything meaningful you found from this or if you found something potentially interesting. Note: You can also use data from other sources for comparison. Explain why the statistics run are meaningful. 
# In the following code blocks, we explore the frequency of Cancelled and Delayed flights. Additionally, we use this section to explore any differences in our continuous features when grouped by the True and False levels of our dependent features.

# In the following code blocks, we explore the frequncy of Cancelled and Delayed flights. Additionaly, we use this section to explore any differences in our continuous features when grouped by the True and False levels of our dependent features.

# In[23]:


# Cancellation Frequency
crit_temp_df = pd.DataFrame(data['critical_temp'].value_counts()).reset_index()
crit_temp_df.columns = ['critical_temp', 'Count']
crit_temp_df['Frequency'] = round(crit_temp_df['Count'] / sum(crit_temp_df['Count']) * 100, 2)
crit_temp_df


# In[21]:


# Histogram of Critical Temperature
fig = px.histogram(crit_temp_df, x="critical_temp", nbins = 20)
fig.show()


# The most interesting detail of note from this table is the overall variation in the distribution of critical temperature. Noting that the range for when a superconductor goes critical can vary greatly between the low of 1 degree celeicus to a high of 144 degrees celicus.  In the little over 21,000 observations from the study, only 143 (0.67%) of superconductors reached critical temperatre at 80 degrees celiecus. There does appear to be some outliers in the data as the histogram reveals a right tailed distribution, we may want to logrithmically scale the data to standardize our analysis.  4155555555555555

# ## Visualize Attributes

# Below we see a histogram of the Average Atomic Mass between the superconductor temperatures achiecving critical mass. We see the average atomic mass rapidly rise as we hit the 72 - 78 marker then drop signficantly at around 80 only to rise again at around 88. A gradual decline with many peaks and valleys until the 144 mark where the peaks begin to become noticeably smaller in size and frequency. The bulk of the data as an average atomic mass between 70 - 90 with fewer values on the outskirts of each.

# In[29]:


# Histogram of Atomic mass
fig = px.histogram(data, x='mean_atomic_mass', marginal="box",width=800, height=400, title='Distribution Plot - Average Atomic Mass')
fig.show()


# Below we see a Histogram of Airtime. This histogram has a similar shape to that of the Distance Histogram but with much smoother peaks and valleys. We see rapped growth until the peak at the 44-45 minuet mark. After this we see a few peaks around the 60-61 and 84-85 minute mark but otherwise see a steady decline downwards to the 170-171 minute mark where the decline begins to smooth out. It is interesting to see how the histograms of distance and air time show the same overall trend but the Airtime trend is much smoother. This may mean that the relationship in distance and airtime can be see when looking at a large number of flights or when there is a great difference in distance traveled but when looking at flights that are within 20 or 30 miles distance traveled of each other there would be a much smaller if any noticeable difference in airtime but further analysis would be needed to make a formal claim.

# In[32]:


# Histogram by AirTime
fig = px.histogram(data, x='wtd_mean_ThermalConductivity', 
                   marginal="box",width=800, height=400, title='Distribution Plot - Thermal Conductivity')
fig.show()


# Below is a Histogram of Elapsed Time. We see that it resembles the trend we have seen in the distance and air time histograms but with a slightly later peak. The peak is at the 82-83 minute mark. We then see a gradual decline downwards until the 204-5 minute mark, once again slightly later than the airtime histogram. This slight delay between airtime and elapsed time makes sense as we are adding all the time spent on the ground to the airtime. It is interesting to see that the growth and decline of the elapsed time is smoother than that of the airtime as there is a notable absence of peaks and valleys in the Elapsed time plot.

# In[ ]:


# Histogram by Elapsed Time
fig = px.histogram(flight_data_df, x='ActualElapsedTime', marginal="box",
                   width=800, height=400, title='Distribution Plot - Elapsed Time')
fig.show()


# Below is a Histogram of departure times of flights that were delayed. We see that late night and early morning flights are rarely delayed and then there is a rapid increase at 06:00. There is steady growth throughout the day until 19:00 where we begin to see a steady decline throughout the remaining hours in the day. This histogram shows that from our sample there is no time during the day where you are significantly less likely to not get delayed but if you fly at night there are far fewer flights that get delayed.

# In[ ]:


# Histogram - Arrival Time of Delayed Flights
fig = px.histogram(flight_data_df[flight_data_df['Delayed']==True], x='DepTime',
                   width=800, height=400, title='Box Plot - Arrival Time of Delayed Flights')
fig.show()


# Below is a plot histogram of arrival times of flights that were delayed. We see a moderate number of flights arrive between midnight and 01:00 and then see a dip until 05:00 when it begins to rise again until the peak at 20:20-20:39. It is interesting to see a similar pattern to that of the departure times of delayed flits just with a 1 to 2 hour delay.

# In[ ]:


# Histogram - Delayed Departure Time of Delayed Flights
fig = px.histogram(flight_data_df[flight_data_df['Delayed']==True], x='ArrTime', 
                   width=800, height=400, title='Box Plot - Arrival Time of Delayed Flights')
fig.show()


# ## Explore Joint Attributes
# Visualize relationships between attributes: Look at the attributes via scatter plots, correlation, cross-tabulation, group-wise averages, etc. as appropriate. Explain any interesting relationships.

# In[ ]:


# First step to explore any relationships between data would be to do a correlation
flight_data_df.corr()


# In[ ]:


# Correlation Plot using Plotly
flight_corr = flight_data_df.corr()

fig = go.Figure()        

fig.add_trace(
    go.Heatmap(
        x = flight_corr.columns,
        y = flight_corr.index,
        z = np.array(flight_corr),
        text=flight_corr.values,
        texttemplate='%{text:.2f}' #set the size of the text inside the graphs

    )
)

fig.update_layout(
    autosize=False,
    width=1000,
    height=600
)

fig.show()


# The correlation plot grants insight into what features might have some type of relationship among themselves. From here I note that CRSDepTime & DepTime have an almost perfect positive relationship indicating to me that only one of these features may be necessary come modeling as they explain the same data. Wheelsoff & CRSArrTime have a strong positive correlation to one another, meanwhile there doesn't appear to be any strong negatively correlated features that represent an inverse relationship to any signficant degree. One other noteable correlation would be Arrival and Departure delay, these features have a strong correlation at ninety-one percent, this stands to reason if the flight is delayed departing it will likely arrive at a delayed time marker. 

# In[ ]:


# Scatter plot of Arr Delay & Dep Delay Minutes by Airline
fig = px.scatter(flight_data_df, x = 'ArrDelay', y='DepDelay', color='Airline',
                 title='Scatterplot - Arrival & Departure Delays by Airline', width=1000, height=600)
fig.update_xaxes(categoryorder='total descending')


fig.show()


# The scatterplot showcases the relationship between Arrival Delays & Departure Delays. The very strong correlation shows a highly correlated display at ninety-six percent. The breakout by the categorical airline indicates that most of the delays are below the 300 minute mark, however American Airlines has some of the highest delays in the dataset overall.  

# In[ ]:


# Box plot of TaxiOut by Airline
fig = px.box(flight_data_df, x = 'Airline', y='TaxiOut', color='Airline', 
             title='Boxplot - Taxiout by Airline', width=1000, height=600)
fig.update_xaxes(categoryorder='total descending')
fig.show()


# We explore the relationshp between airlines and there distribution of time it taxes to taxi out to the runway. Some noteable measurements during the year 2021 based on our sample population we see taht Empire Airlines has the tightest distribution with almost no outliers indicating that delays are not a prevelant thing for them. Meanwhile of the major US airlines; JetBlue, American, Delta, United, & Southwest we note that American has the largest distribution gap with a median taxi out time of approximately fifteen minutes while Southwest has a median taxi out time of approximately eleven minutes.

# In[ ]:


# Scatter plot of Arr Time  & Distance  
fig = px.scatter(flight_data_df, x = 'AirTime', y='Distance', color = 'Airline', 
                 title='Scatterplot - Air Time and Distance', width=1000, height=600)
fig.update_xaxes(categoryorder='total descending')
fig.show()


# The scatterplot showcases the relationship between Air Time & Distance. The nearly perfectcorrelation at ninety-eight percent indcates the relationship that the longer a plane travels the more distance it travels. What becomes interesting however is that Hawaiian Airlines Inc has air time around the three hundred to four hundred minute mark but then at the upper most marker around 550 minutes with other notable airlines for european travel clustered between them. 

# In[ ]:


#Create new Df for graphing Hawaiian Flights
hawaiian_airlines = flight_data_df.query('Airline == "Hawaiian Airlines Inc."')
hawaiian_airlines.head() #head filter function


# In[ ]:


# Scatter plot of Hawaiian AIrlinges for air time 
fig = px.scatter(flight_data_df, x = 'AirTime', y='Distance', color = 'OriginState',
                 title='Scatterplot - Air Time and Distance for Hawaiian Airlines', width=1000, height=600)
fig.update_xaxes(categoryorder='total descending')
fig.show()


# Wanting to get a deeper understanding of where some of those outliers of very high Air Time are present for Hawaiian Airlines, the scatterplot above is a focused view from the aggregate data above. We see that those long air time flights are primarily coming from the east coast directly to Hawaii for their end destination. Some of the higher notable mentions we see a number of Massacheutts, Florida, and New York flights being among the highest air time and distance traveled per the population of the dataset. 
# 

# In[ ]:


# Histogram plot of DestState by Diverted
fig = px.bar(flight_data_df, x = 'DestState', y='Diverted', 
             title='Amount of flights diverted by final Destination', width=1000, height=600)
fig.update_xaxes(categoryorder='total descending')
fig.show()


# We examine the relationshp between the end destination and whether a flight was diverted from its original destintation. In that respect we can clearly see that flights bound to Texas and Flordia have the highest degree of diversions compared to any other end destination in the continental US. 

# In[ ]:


# Box plot of TaxiOut by Airline
fig = px.box(flight_data_df, x = 'OriginState', y='DepDelayMinutes', 
             title='Boxplot - Origin State Departure Delays', width=1000, height=600)
fig.update_xaxes(categoryorder='total descending')
fig.show()


# We wanted to examine based on the population if the origin state of the departure flight has any impact on departure delays for that state. There does seem to be noteble instances where certain states have higher percentage of delays based on their point of origin. Being Texas, Florida, and California having the largest number of departure delays present then the rest of the states where flights depart from. 

# ## Explore Attributes and Class
# Identify and explain interesting relationships between features and the class you are trying to predict (i.e., relationships with variables and the target classification).

# In this section we want to explore the relationship of minutes delayed when grouping by airline, month, and day of the month. Until we complete the regression and classification portion of our analysis, we won't be able to give hard numbers into the weight of each of these categorical variables. However, this section can show us some fast information about airlines, months, or days of the month to avoid flying.

# In[ ]:


# Box plot of Log Delay Minutes by Airline
fig = px.box(flight_data_df[flight_data_df['Delayed'] == True], x = 'Airline', y='LogDepDelayMinutes', color='Airline',
             title='Boxplot - LogDepDelayMinutes by Airline', width=1000, height=600)
fig.update_xaxes(categoryorder='total descending')
fig.show()


# For this plot, we decided to go ahead and use LogDepDelayMinutes. Without doing so, we would be unable to see any differences between the groups due to the positive skewness of the feature.
# 
# This plot is sorted by overall count of flights in the sample. Not surprisingly, we see the big airlines crowded at the front of the table. Of the top 5 in flight volume, Delta Air Lines has a noticeably lower mean and interquartile range.
# 
# In the next plot, we grouped the top airlines and compared them to the budget airlines and the rest of the pack. Cheap airlines were references using an article on www.lendedu.com that discusses, a now unavailable, study on www.rome2rio.com (Serpette, 2021). As expected, the "Budget" airlines are visibly higher in minutes delayed.

# In[ ]:


# Comparing big name airlines verse budget airlines
big_name =['Southwest Airlines Co.', 'American Airlines Inc.', 'Delta Air Lines Inc.', 'United Air Lines Inc.',
           'SkyWest Airlines Inc.']
budget = ['Spirit Air Lines', ' Frontier Airlines Inc.', 'Hawaiian Airlines', 'Allegiant Air', 'Sun Country Airlines']

flight_data_df['AirlineGroup'] = np.where(flight_data_df['Airline'].isin(big_name), 
                                           'Big Name', np.where(flight_data_df['Airline'].isin(budget), 'Budget', 'No Group'))

fig = px.box(flight_data_df[flight_data_df['Delayed'] == True],
             x = 'AirlineGroup', y='LogDepDelayMinutes', color='AirlineGroup', 
             title='Boxplot - LogDepDelayMinutes by Airline', width=1000, height=600)
fig.update_xaxes(categoryorder='total descending')
fig.show()


# The boxplot of months is a little difficult to make interpretations from. However, there appears to be an up and down flow throughout the year. In order to see the differences more easily, we created a line plot to view the mean LogDepDelayMinutes from each month.

# In[ ]:


# Log Delay Minutes by Month
fig = px.box(flight_data_df, x = 'Month', y = 'LogDepDelayMinutes', 
             title='Boxplot - LogDepDelayMinutes by Month', width=1000, height=600)
fig.show()


# In[ ]:


# Mean LogDepDelayMinutes grouped by Month
month_df = flight_data_df[flight_data_df['Delayed'] == True].groupby('Month')[['LogDepDelayMinutes']].mean().round(2).reset_index()
fig = px.line(month_df, x = 'Month', y = 'LogDepDelayMinutes',
              title='Mean LogDepDelayMinutes by Month', width=1000, height=600)
fig.show()


# This plot clearly shows the trend in LogDepDelayMinutes throughout the year. My personal expectation was to see higher delays during the holidays. Instead, we are seeing evidence that supports the summer months of June, July, and August being the highest.
# 
# For Day of Month, we skipped the box plot and went straight to the line plot. The trend is not nearly as obvious when looking at the day of the month. See as the days of the month fall on different days of the week throughout the year, this is not a surprising result.

# In[ ]:


# Mean LogDepDelayMinutes grouped by Month
day_df = flight_data_df[flight_data_df['Delayed'] == True].groupby('DayofMonth')[['LogDepDelayMinutes']].mean().round(2).reset_index()
fig = px.line(day_df, x = 'DayofMonth', y = 'LogDepDelayMinutes', title='Mean LogDepDelayMinutes by Dat',
              width=1000, height=600)
fig.show()


# In these last couple plots, we explore the feature Distance. In the Simple Statistics section of our analysis, we were able to see a much higher mean Distance for Delayed flights. This leads us to believe it will be an important feature in both our regression and classification models. For that reason, we wanted to use this space to briefly explore the Distance outliers.
# 
# Our first approach to handle these outliers would be to replace their values with something more representative with the rest of the model. In order to safely do this, we decided to look at the distribution plot and scatter plots. The goal here is to see if there is any linear correlation between Distance and DepDelayMinutes at the top range of Distance. From the Distribution plot, we can clearly see the Delayed True group has a higher mean, but the outliers appear to be grouped with one another. This nearly one for one patter of the outliers is likely due to another feature relationship that we haven't discovered yet.
# 
# In the scatter plot, we see no linear relationship between Distance and DepDelayMinutes. If anything, it appears there might be some quadratic relationship. Either way, there is no immediately obvious evidence that suggests we can't replace the outliers with a value more representative of the sample (e.g., Upper Quartile). Once again, we will have to evaluate the effects of this potential transformation or imputing in the regression and classification part of the analysis. 

# In[ ]:


# Delayed Status Histogram by Distance
fig = px.histogram(flight_data_df, x='Distance', color='Delayed', marginal="box",
                   width=800, height=400, title='Distribution Plot - Delayed Status by Distance')
fig.show()


# In[ ]:


# Delayed Status vs Distance
fig = px.scatter(flight_data_df, x='Distance', y='DepDelayMinutes', color='Delayed', facet_col='Delayed',
                 width=800, height=400, title='Delayed Status Histogram by Distance')
fig.show()


# For our last feature of this section, we chose DepTime. It showed strong evidence to suggest greater departure times resulted in delayed flights. To show this, we've created another distribution plot. The plot shows much higher values at every quartile for the delayed flights and the relative frequency of delayed flights appears to be much higher as Departure Time increases.

# In[ ]:


# Delayed Status Histogram by Departure Time
fig = px.histogram(flight_data_df, x='DepTime', color='Delayed', marginal="box",
                   width=800, height=400, title='Distribution Plot - Status Histogram by Departure Time')
fig.show()


# ## New Features
# Are there other features that could be added to the data or created from existing features? Which ones?
# 
# Throughout our initial, we created Delayed, AirlineGroup, and several DepDelayMinutes transformations. These features alone, allowed us to see underlying trends in the data that were not priorly available. In addition to these features, we would love to find an external source that allowed us to implent a cost ellement to our dataset.
# 
#  - Average airline cost / minute delayed
#  - Average passenger cost / minute delayed
#  - Average reimbursement / cancellation
# 
# Below are the three DepDelayMinutes transformations we implemnted and will be testing in the next phase of this project.

# In[ ]:


# Log Transformation
fig = px.histogram(flight_data_df[flight_data_df['Delayed'] == True], x = 'LogDepDelayMinutes',
                   width=800, height=400, marginal="violin",title="Distplot - Log Departure Delay Time")
fig.show()


# In[ ]:


# Square Root Transformation
fig = px.histogram(flight_data_df[flight_data_df['Delayed'] == True], x = 'SqrtDepDelayMinutes',
                   width=800, height=400, marginal="violin",title="Distplot - Sqrt Departure Delay Time")
fig.show()


# In[ ]:


# Cube Root Transformation
fig = px.histogram(flight_data_df[flight_data_df['Delayed'] == True], x = 'CubrtDepDelayMinutes',
                   width=800, height=400, marginal="violin",title="Distplot - Qubrt Departure Delay Time")
fig.show()


# # Modeling

# # Exceptional Work

# In[ ]:


##Multi-Nomial Regression with Distance Group/ EXCEPTIONAL WORK

df3= flight_data_df

df4 = df3.select_dtypes(include = ['float','integer'])

#Removing NaN Values
for column in df4.columns:
    if df4[column].isnull().any():
        count = df4[column].isnull().sum()
        print(column + " has " +str(count)+" NaN values")
       
df3_reduced = df4.dropna()

# Dropping columns with less than 2 classes

df3_new = df3_reduced.loc[:, df3_reduced.apply(pd.Series.nunique) > 1]


#df3_new = df3_new.select_dtypes(include = )
#print(X2.dtypes)
X = df3_new.drop('DistanceGroup', axis = 1)
y = df3_new.DistanceGroup


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state = 50)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


#Scaling the data
#Credit to: https://scikit-learn.org/stable/modules/preprocessing.html
#scaler = preprocessing.StandardScaler().fit(X_train)

#X_scaled = scaler.transform(X)

X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#Credit to https://machinelearningmastery.com/multinomial-logistic-regression-with-python/
pipe = make_pipeline(StandardScaler(), LogisticRegression(multi_class = 'multinomial',solver = 'lbfgs'))
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
pipe.fit(X_train, y_train)  # apply scaling on training data
Pipeline(steps=[('standardscaler', StandardScaler(with_mean=False)),
                ('logisticregression', LogisticRegression())])


final = pipe.score(X_test, y_test)
print('Accuracy with No L2 Penalty')
print('Accuracy with Standardization and Cross Validation: %.3f' % (mean(n_scores)))
print('Accuarcy with Standardization and not Cross Validation: %.3f' % (final))
#Creating Confusion Matrix
#Credit to https://realpython.com/logistic-regression-python/
confusion_matrix(y,pipe.predict(X))
cm = confusion_matrix(y, pipe.predict(X))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Distance Group 0', 'Predicted Distance Group 1'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Distance Group 0', 'Actual Distance Group 1'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


#Creating Output from Models
print(classification_report(y, pipe.predict(X)))

#With L2 Penalty
pipe = make_pipeline(StandardScaler(), LogisticRegression(multi_class = 'multinomial',solver = 'lbfgs', penalty = 'l2',C=.05))
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty = 'l2',C=.05)
n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
pipe.fit(X_train, y_train)  # apply scaling on training data
Pipeline(steps=[('standardscaler', StandardScaler(with_mean=False)),
                ('logisticregression', LogisticRegression())])


final = pipe.score(X_test, y_test)
print('Accuracy with L2 Penalty')
print('Accuracy with Standardization and Cross Validation: %.3f' % (mean(n_scores)))
print('Accuarcy with Standardization and not Cross Validation: %.3f' % (final))

#Creating Confusion Matrix
#Credit to https://realpython.com/logistic-regression-python/
confusion_matrix(y,pipe.predict(X))
cm = confusion_matrix(y, pipe.predict(X))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Distance Group 0', 'Predicted Distance Group 1'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Distance Group 0', 'Actual Distance Group 1'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


#Creating Output from Models
print(classification_report(y, pipe.predict(X)))


# From the output above we can see the logistic regression for classifying Distance group is highly accurate. Looking at the Confusion Matrix below, we see there is a difference not only with the implementation of K-Fold Cross Validation, but also with the addition of the L2 Penalty. We are more confident in the models without the L2 penalty as these classify more accurately as well as provide enough room to provide post model adjustments if need be. This is our first attempt at modeling with our Flight dataset and in the future will bring about more robust methods to our modeling. Upon further review of the model, there are many more directions available for us to entertain, but would require a more in depth understanding of modeling within Pandas.

# # Sources
# 1. Ball, M., Barnhart, C., Dresner, M., Hansen, M., Neels, K., Odoni, A., Peterson, E., Sherry, L., Trani, A., & Zou, B. (2010, October). Total Delay Impact Study. isr.umd.edu. https://isr.umd.edu/NEXTOR/pubs/TDI_Report_Final_10_18_10_V3.pdf
# 
# 2. Olson, E. (2022, December 28). Soutwest cancels another 4,800 flights as its reduced schedule continues. NPR. https://www.npr.org/2022/12/28/1145775020/southwest-airlines-flights-cancellations
# 
# 3. Investis. (2023, January 6). Southwest Airlines Securities and Exchange Commission Financial Filing. https://otp.tools.investis.com/clients/us/southwest/SEC/sec-show.aspx?Type=html&FilingId=16303107&CIK=0000092380&Index=10000
# 
# 4. Serpette, S. (2021, June 16). 10 Cheapest Airlines to Fly With: See the Results. LendEDU. https://lendedu.com/blog/10-cheapest-airlines-to-fly-with
# 
# 5. 6.3. Preprocessing data. (n.d.). Scikit-learn. https://scikit-learn.org/stable/modules/preprocessing.html
# 
# 6. Real Python. (2022a, September 1). Logistic Regression in Python. https://realpython.com/logistic-regression-python/
# 
# 7. Machine Learning Mastery (n.d.). https://machinelearningmastery.com/multinomial-logistic-regression-with-python/
