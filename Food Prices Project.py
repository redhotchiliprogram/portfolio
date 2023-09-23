#!/usr/bin/env python
# coding: utf-8

# # India Food Prices Project

# ## Setup

# In[1]:


import pandas as pd
import numpy as np
df_food = pd.read_csv('wfp_food_prices_ind.csv')
df_food.head()


# ## Cleaning the dataset

# In[2]:


df_food.info()


# In[3]:


df_food = df_food.loc[1:, :]
df_food.head()


# In[4]:


df_food['priceflag'].value_counts()


# In[5]:


df_food['currency'].value_counts()


# Some basic problems with the dataset so far:
# 1. Redundant columns like `priceflag` and `currency` that contain only one unique value. Need to be removed.
# 2. All columns have `object` dtype, need to be corrected to most appropriate dtype.
# 3. Some columns would be more useful if renamed For example: `admin1`, `admin2`, `price`.
# 4. We are only interested in price in INR, the domestic currency, so the `usdprice` column becomes redundant.
# 5. `latitude` and `longitude` are not relevant to our analysis.
# 6. Dates are not uniform across states--for some states the earliest records are 1994, for some they are 2012.

# In[6]:


#Correcting dtypes
df_food['admin1'] = df_food['admin1'].astype('category')
df_food['admin2'] = df_food['admin2'].astype('category')
df_food['market'] = df_food['market'].astype('category')
df_food['latitude'] = df_food['latitude'].astype('float64')
df_food['longitude'] = df_food['longitude'].astype('float64')
df_food['category'] = df_food['category'].astype('category')
df_food['commodity'] = df_food['commodity'].astype('category')
df_food['unit'] = df_food['unit'].astype('category')
df_food['priceflag'] = df_food['priceflag'].astype('category')
df_food['pricetype'] = df_food['pricetype'].astype('category')
df_food['price'] = df_food['price'].astype('float64')
df_food['usdprice'] = df_food['usdprice'].astype('float64')


# In[7]:


#drop 'currency', 'priceflag', 'usdprice'
df_food = df_food.drop(['priceflag', 'currency', 'usdprice', 'latitude', 'longitude'], axis=1)

#rename 'admin1', 'admin2', and 'price'
df_food = df_food.rename({'price':'inrprice', 'admin1':'state', 'admin2':'district'}, axis=1)


# In[8]:


df_food.sort_values(by = 'date')


# In[9]:


#Checking for null values
print(df_food['state'].isna().sum())
df_food['district'].isna().sum()


# In[10]:


#Dropping null values from our df (only missing values are in 'state' and 'district' columns,)
df_food = df_food[pd.notnull(df_food['state']) & pd.notnull(df_food['district'])]
df_food.isna().sum().sum()


# In[11]:


df_food[df_food['unit']=='L'].sort_values(by='date')


# In[12]:


df_food[df_food['unit']=='KG'].sort_values(by='date')


# In[13]:


bool_mask1 = df_food['unit'] == '100 KG'
bool_mask2 = df_food['pricetype'] == 'Retail'
df_food[bool_mask1 & bool_mask2]


# In[14]:


bool_mask3 = df_food['unit'] == 'KG'
df_food[bool_mask3 & ~bool_mask2]


# In[15]:


bool_mask4 = df_food['unit'] == 'L'
df_food[bool_mask4 & ~bool_mask2] #36


# There are three different measures of weight, for different products and over different time periods. This complicates things, and we should either standardise or drop those values that we don't need.
# 
# We can see that `100 KG`  corresponds exclusively to pricetype `wholesale`. Meanwhile, both `L` and `KG` correspond to `retail` pricetype. This information will be useful when wrangling the data.

# ## Data Wrangling

# Having done some initial cleaning, we can now wrangle the dataset to prepare it for further analysis.
# 
# Questions we can ask:
# 1. What states had highest/lowest food inflation since 1994?
# 2. What states had highest/lowest food inflation in the post-pandemic opening-up?
# 3. The above two questions, but for categories and commodities of food instead of states
# 4. Answering all above questions separately for wholesale and retail inflation
# 
# To answer these questions, need to first complete some tasks:
# 1. Create separate dfs for retail and wholesale calculations
# 2. Creating a new df with yearly indices of inflation: need to create a basket of goods to calculate price change with
# 3. Editing the df such that we have a uniform timeframe across states/UTs to work with

# In[16]:


df_food['state'].value_counts()


# Not all states have data up until 1994. We need to find a better starting point that encapsulates more (if not all) states. This has the added benefit of making the analysis more contemporary. For this, it would help to isolate `year` from the `date`.

# In[17]:


df_food['year'] = df_food['date'].str.split(pat='-', expand=True)[0]


# In[18]:


df_food = df_food.drop(['date'], axis=1)


# In[19]:


df_food.head()


# Apart from Chhattisgarh, Sikkim, Manipur, we have data on all states/Union Territories from 2012 onwards. For convenience, let us use 2014 as a reference point for analysis, coinciding with the beginning of the present government's first tenure.
# 
# To further simplify the data for this purpose, let us exclude the above three states from the dataframe.

# In[20]:


cg_filter = df_food['state'] == 'Chhattisgarh'
sk_filter = df_food['state'] == 'Sikkim'
mn_filter = df_food['state'] == 'Manipur'
df_food = df_food[~cg_filter & ~sk_filter & ~mn_filter]
df_food['state'].value_counts()


# In[21]:


#Removing pre-2014 data
df_food['year'] = df_food['year'].astype('int64')
df_food = df_food[df_food['year'] >= 2014]
df_food['year'].describe()


# Taking stock: so far we have cleaned our dataset, and among other things, removed null values and made our timeframe uniform across states. Now our entire dataset represents prices of commodities across states (excluding Chhatisgarh, Sikkim, and Manipur) in the timeframe 2014-2022.
# 
# Now we proceed to separate our analyses for retail and wholesale inflation.

# In[22]:


#New dfs for retail and wholesale inflation
df_retail = df_food[df_food['pricetype'] == 'Retail']
df_wholesale = df_food[df_food['pricetype'] == 'Wholesale']


# In[23]:


df_retail['state'].value_counts()


# In[24]:


df_wholesale['state'].value_counts()


# We see that while retail data is available for all states (excluding Chhattisgarh, Sikkim, Manipur, as discussed above), wholesale data is restricted to only four states. This is inadequate to conduct a meaningful analysis of wholesale prices.

# ## Data Analysis

# For simplification, the index we use will just be an average of all commoities, without any weights attached.

# ### Retail Inflation

# In[25]:


#Step 1: Creating price baskets for each state per year
df_retail_inflation = df_retail.groupby(['state', 'year'])['inrprice'].mean().reset_index()
df_retail_inflation = df_retail_inflation.rename({'inrprice':'price_basket'}, axis=1)
df_retail_inflation.info()


# In[26]:


df_retail_inflation.head()


# In[27]:


#Checking whether the above groupby method yields the correct values
df_test = df_retail[(df_retail['state']=='Andaman and Nicobar') & (df_retail['year']==2014)]
df_test['inrprice'].describe()
#Success! The mean generated by calling describe() is equal to the mean calculated above (see row 1 of above df)


# In[28]:


df_retail_inflation['previous_year_price'] = df_retail_inflation['price_basket'].shift(1)
df_retail_inflation.head()


# In[29]:


#Measuring year on year inflation
df_retail_inflation['yoy_inflation'] = df_retail_inflation['price_basket']-df_retail_inflation['previous_year_price']
df_retail_inflation['yoy_inflation'] = df_retail_inflation['yoy_inflation']/df_retail_inflation['previous_year_price']*100


# In[30]:


df_retail_inflation = df_retail_inflation.drop(['previous_year_price'], axis=1)


# In[31]:


df_retail_inflation['state'].value_counts()


# In[32]:


df_retail_inflation.head(10)


# A problem we  encounter here is the prices of different states being compared together. For example, in row 9, the Andhra Pradesh 2014 price has been compared with the Andaman and Nicobar price of 2022, which gives us a redundant value for inflation in Andhra Pradesh in 2014. We need to drop all such values.

# In[33]:


#Substitute all 2014 inflation values with null values
df_retail_inflation.loc[df_retail_inflation['year']==2014, 'yoy_inflation'] = np.nan
df_retail_inflation[df_retail_inflation['year']==2014]['yoy_inflation'].describe()


# In[34]:


#Calculating total inflation from 2014-2022
df_retail_inflation['2014_price'] = df_retail_inflation['price_basket'].shift(8)
df_retail_inflation['total_inflation'] = df_retail_inflation['price_basket']-df_retail_inflation['2014_price']
df_retail_inflation['total_inflation'] = df_retail_inflation['total_inflation']/df_retail_inflation['2014_price']*100
df_retail_inflation.head(10)


# In[35]:


df_retail_inflation = df_retail_inflation.drop(['2014_price'], axis=1)


# In[36]:


#Dropping redundant values and rounding all values
df_retail_inflation.loc[df_retail_inflation['year']!=2022, 'total_inflation'] = np.nan
df_retail_inflation['price_basket'] = df_retail_inflation['price_basket'].round(2)
df_retail_inflation['yoy_inflation'] = df_retail_inflation['yoy_inflation'].round(2)
df_retail_inflation['total_inflation'] = df_retail_inflation['total_inflation'].round(2)
df_retail_inflation.head(18)


# In[37]:


#Sorting from lowest to highest inflation
df_retail_inflation.sort_values(by='total_inflation', ascending=True).head()


# In[38]:


#Sorting from highest to lowest inflation
df_retail_inflation.sort_values(by='total_inflation', ascending=False).head()


# We have succesfully conducted the first piece of this analysis! As per the data, Assam had the highest total retail food inflation (116.3%) in the 2014-22 period, while Andaman and Nicobar Islands had the lowest (45.67%).

# ### Average retail prices over time

# In[39]:


df_retail_avg = df_retail.groupby('state')['inrprice'].mean().reset_index()
df_retail_avg #Keep this, useful to compare aggregate prices across states over time
df_retail_avg = df_retail_avg.rename({'inrprice':'avginrprice'}, axis=1) #renaming inrprice column
df_retail_avg['avginrprice'] = df_retail_avg['avginrprice'].round(2)
df_retail_avg.head()


# In[40]:


df_retail_avg.sort_values(by='avginrprice', ascending=True).head()


# In[41]:


df_retail_avg.sort_values(by='avginrprice', ascending=False).head()


# Our analysis now tells us that the highest average prices in the 2014-22 period were in the Andaman and Nicobar Islands, while the lowest were in Assam. This is an interesting result, considering we observed earlier that the Andaman and Nicobar Islands had the *lowest* inflation rate and Assam the *highest*. One can probably attribute these inflation rates to base rate effects.

# ### Commodity-wise prices

# Now, on to the next section of the analysis. We shift our focus from state-wise analysis to commodity-wise analysis. Here, we answer the question: Which categories and commodities had the highest/lowest rates of retail inflation in the given time period?

# In[42]:


df_retail.head()


# In[43]:


#Grouping food category prices by year
df_category = df_retail.groupby(['category', 'year'])['inrprice'].mean().reset_index()
df_category = df_category.rename({'inrprice':'avg_national_price'}, axis=1)
print(df_category.info())


# In[44]:


#Checking for uniformity in category distribution
df_category['category'].value_counts()


# In[45]:


#Calculating y-o-y inflation similar to our state-wise analysis:
df_category['prev_year_price'] = df_category['avg_national_price'].shift(1)
df_category['yoy_inflation'] = df_category['avg_national_price']-df_category['prev_year_price']
df_category['yoy_inflation'] = df_category['yoy_inflation']/df_category['prev_year_price']*100
df_category.loc[df_category['year']==2014, 'yoy_inflation'] = np.nan #Removing redundant values
df_category.head(10)


# In[46]:


df_category = df_category.drop(['prev_year_price'], axis=1)


# In[47]:


#Total inflation
df_category['2014_price'] = df_category['avg_national_price'].shift(8)
df_category.loc[df_category['year']!=2022, '2014_price'] = np.nan
df_category['total_inflation'] = df_category['avg_national_price']-df_category['2014_price']
df_category['total_inflation'] = df_category['total_inflation']/df_category['2014_price']*100
df_category.head(10)


# In[48]:


#Cleaning up
df_category = df_category.drop(['2014_price'], axis=1)
df_category['avg_national_price'] = df_category['avg_national_price'].round(2)
df_category['yoy_inflation'] = df_category['yoy_inflation'].round(2)
df_category['total_inflation'] = df_category['total_inflation'].round(2)


# In[49]:


df_category


# In[50]:


df_category.sort_values(by='total_inflation', ascending=True).head()


# In[51]:


df_category.sort_values(by='total_inflation', ascending=False).head()


# We see that the highest price rise in the 2014-22 period was witnessed by pulses and nuts (80.22%), while the lowest was for the category miscellaneous food (14.07%).

# ### The post-pandemic inflation

# Let us also use the wrangled dataset for insights on retail inflation as per states and food categories in the context of the post-pandemic inflation.
# 
# In 2020, the government of India imposed a nationwide lockdown to prevent the virus from spreading, and it was only later that year that the lockdown was lifted. The lockdown period was accompanied by deflation in prices, and subsequently, the post-lockdown period was a high-inflation one, owing to the rebound of economic activity and demand. This inflation has been persistent till date (August 2023), and so it would be pertinent to measure food retail inflation in the years 2020-22 (the latest year till which we have data).
# 
# The method for doing this will be the exact same as our above two analyses for state-wise and category-wise inflation.

# In[52]:


#State-wise analysis
df_post_covid = df_retail_inflation[df_retail_inflation['year']>=2020]
df_post_covid.head(10)


# In[53]:


#Redefining total_inflation
df_post_covid['2020_price'] = df_post_covid['price_basket'].shift(2)
df_post_covid['total_inflation'] = df_post_covid['price_basket']-df_post_covid['2020_price']
df_post_covid['total_inflation'] = df_post_covid['total_inflation']/df_post_covid['2020_price']*100
df_post_covid.head()


# In[54]:


df_post_covid = df_post_covid.drop(['2020_price'], axis=1)


# In[55]:


#Cleaning up
df_post_covid.loc[df_post_covid['year']!=2022, 'total_inflation'] = np.nan
df_post_covid['total_inflation'] = df_post_covid['total_inflation'].round(2)
df_post_covid.head(6)


# In[56]:


df_post_covid.sort_values(by='total_inflation', ascending=True).head()


# In[57]:


df_post_covid.sort_values(by='total_inflation', ascending=False).head()


# In the post-pandemic inflation period, Tripura saw the most severe retail food price rise (38.07%), while Mizoram saw the least (8.05%).

# In[58]:


#Category-wise analysis
df_post_covid_category = df_category[df_category['year'] >= 2020]
df_post_covid_category.head()


# In[59]:


#Redefining total inflation
df_post_covid_category['2020_price'] = df_post_covid_category['avg_national_price'].shift(2)
df_post_covid_category.loc[df_post_covid_category['year']!=2022, '2020_price'] = np.nan
df_post_covid_category['total_inflation'] = df_post_covid_category['avg_national_price']-df_post_covid_category['2020_price']
df_post_covid_category['total_inflation'] = 100*df_post_covid_category['total_inflation']/df_post_covid_category['2020_price']
df_post_covid_category['total_inflation'] = df_post_covid_category['total_inflation'].round(2)
df_post_covid_category.head(6)


# In[60]:


df_post_covid_category = df_post_covid_category.drop(['2020_price'], axis=1)


# In[61]:


df_post_covid_category.sort_values(by='total_inflation', ascending=True).head()


# In[62]:


df_post_covid_category.sort_values(by='total_inflation', ascending=False).head()


# Finally, we see that in the post-pandemic period, oil and fats saw the highest price rise (47.36%), while the lowest inflation was that of vegetables and fruits, which was in fact a price fall of -13.63%.
