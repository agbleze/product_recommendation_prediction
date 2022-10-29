
#%%
import os
import pandas as pd


# %%
cwd_path = os.getcwd()


# %%
data_path = cwd_path + '/data/product_reviews.csv'


df = pd.read_csv(data_path)


#%%

df.columns

#%%
review_df =  df[['reviews.text', 'reviews.doRecommend']]

#%%
review_df['reviews.doRecommend'].value_counts()


#%%
review_df.dropna()


#%%
review_df_clean = review_df[review_df['reviews.text']!='Rating provided by a verified purchaser'].dropna()



# %%
review_df_clean['reviews.doRecommend'].value_counts()

#%%
from sklearn.model_selection import train_test_split


#%%

df_train, df_test = train_test_split(review_df_clean, train_size=0.7, #stratify='reviews.doRecommend',
                 random_state=0)


#%%


df_train.info()



# %%
