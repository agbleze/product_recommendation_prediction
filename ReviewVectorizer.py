#%%
from Vocabulary import ReviewSequenceVocabulary, ReviewVocabulary
import numpy as np
import pandas as pd
import string

#%%
def preprocess_texts_to_tokens(sentence):
    text_tokens = []
    for text in sentence.split(" "):
            text = text.lower()
            if text not in string.punctuation and not text.isnumeric():
                if text[-1] in string.punctuation:
                    text = text[:-1]
                text_tokens.append(text)
    return text_tokens
    
sent= "this is a Great proDUCT?"


for token in preprocess_texts_to_tokens(sent):
    print(token)

#print(res)


#print(sent_list)

#%%
class ReviewVectorizer:
    def __init__(self, review_df):
        self.review_df = review_df
        
        self.review_vocab = ReviewSequenceVocabulary()
        self.recommend_vocab = ReviewVocabulary()
        
    def vectorize(self, review_text, vector_length = -1):
        
        indices = [self.review_vocab.begin_seq_index]
        indices = indices.extend(self.review_vocab.lookup_token(token) for token in preprocess_texts_to_tokens(review_text))
        indices = indices.append(self.review_vocab.mask_index)
        
        if vector_length >= 0:
            outer_vector = np.empty(len(indices), dtype=np.int64)
            outer_vector[:len(indices)] = indices
            outer_vector[len(indices):] = self.review_vocab.mask_index
            return outer_vector
        
    def get_vectorizer(self):
        return self.vectorize(vector_length=-1)
    
    
    #@classmethod
    def from_dataframe(self):
        #cls.review_df = review_df
        for row, text_column in self.review_df.iterrows():
            for token in preprocess_texts_to_tokens(text_column['reviews.text']):
                self.review_vocab_token = self.self.review_vocab.add_token(token) 
        
        for token in sorted(set(preprocess_texts_to_tokens(self.review_df['reviews.doRecommend']))):
            self.recommend_vocab.add_token(token)
            
        return (self.review_vocab_token, self.recommend_vocab)
        
        
