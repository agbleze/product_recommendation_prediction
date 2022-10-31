
#%%
import os
import pandas as pd
from nltk import tokenize, word_tokenize
from nltk.book import FreqDist
import numpy as np

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

df_train, df_smallset = train_test_split(review_df_clean, train_size=0.7, #stratify='reviews.doRecommend',
                 random_state=0)


#%%

df_train.info()

#%%

df_train['reviews.doRecommend'].value_counts().plot(kind='bar')

# %%

df_smallset['reviews.doRecommend'].value_counts().plot(kind='bar')

#%%

df_test, df_val = train_test_split(df_smallset, train_size=0.5, random_state=0)


#%%
from collections import Counter
import string



#%%

word_counter = Counter()

#%%
for review in df_train['reviews.text']:
    for text in review.split(" "):
        text = text.lower()
        if text not in string.punctuation and not text.isnumeric():
            if text[-1] in string.punctuation:
                text = text[:-1] 
            word_counter[text] += 1
            
#%%    
wd = Counter()
for w in text.split(" "):
    word = word.lower()
    if w not in string.punctuation:
        if w[-1] in string.punctuation:#[',', '.']:
            w = w[:-1]
        wd[w] += 1
print(wd)


#%% token to index
token_to_idx = {}

for token in word_counter:
    idx = len(token_to_idx)
    token_to_idx[token] = idx

#%% idx_to_token
idx_to_token = {}

for token, idx in token_to_idx.items():
    idx_to_token[idx] = token

#%%
def add_token(token: str):
    if token in token_to_idx:
        idx = token_to_idx[token] 
    else:
        idx = len(token_to_idx)
        token_to_idx[token] = idx
        idx_to_token[idx] = token
    return idx
        
#%%
add_token('work')        
        

#%% provide constant for various token
mask_token = "<MASK>"
begin_seq_token = "<BEGIN>"
end_seq_token = "<END>"
unk_token = "<UNK>"

#%%
indices = [add_token(begin_seq_token)]
indices.extend(add_token(token) for token in 'I love the product'.split(" "))
indices.append(add_token(end_seq_token))

#%%
vector_length = len(indices)
outer_vector = np.zeros(vector_length)

#%%
outer_vector[:vector_length] = indices
#outer_vector[vector_length] = add_token(mask_token)


#%%

train_freq = FreqDist(df_train['reviews.text'])



#%%
text = 'Bought two for the. added comfort they. provide and. the wide straps. I am very pleased with my purchases and will probably order another two as they fit so well and provide full support.'

for word in text.split(" "):
    word = word.lower()
    if word[-1] in [',', '.']:
        word = word[:-1]
    print(word)


#%%
wd = Counter()
for w in text.split(" "):
    word = word.lower()
    if w not in string.punctuation:
        if w[-1] in string.punctuation:#[',', '.']:
            w = w[:-1]
        wd[w] += 1
print(wd)






# %%
