#%%
from Vocabulary import ReviewSequenceVocabulary, ReviewVocabulary
import numpy as np
import pandas as pd
import string
import json

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
    def __init__(self, review_df: pd.DataFrame):
        self.review_df = review_df
        
        self.review_vocab = ReviewSequenceVocabulary()
        self.recommend_vocab = ReviewVocabulary()
        
        tokenize_text_length = lambda content: len(content.split(" "))
        self._max_seq_length = max(map(tokenize_text_length, self.review_df['reviews.text'])) + 2
        
        
    def vectorize(self, review_text, vector_length = -1):
        
        indices = [self.review_vocab.begin_seq_index]
        indices = indices.extend(self.review_vocab.lookup_token(token) for token in preprocess_texts_to_tokens(review_text))
        indices = indices.append(self.review_vocab.mask_index)
        
        if vector_length >= 0:
            outer_vector = np.empty(self._max_seq_length, dtype=np.int64)
            outer_vector[:len(indices)] = indices
            outer_vector[len(indices):] = self.review_vocab.mask_index
            return outer_vector
        
    def get_vectorizer(self):
        return self.vectorize(vector_length=-1)
    
    def save_vectorizer(self, filepath: str):
        self.filepath = filepath
        vectorizer = self.get_vectorizer()
        json.dump(vectorizer, self.filepath)
        
    
    
    #@classmethod
    def from_dataframe(self):
        #cls.review_df = review_df
        for row, text_column in self.review_df.iterrows():
            for token in preprocess_texts_to_tokens(text_column['reviews.text']):
                self.review_vocab_token = self.review_vocab.add_token(token) 
        
        for token in sorted(set(preprocess_texts_to_tokens(self.review_df['reviews.doRecommend']))):
            self.recommend_vocab.add_token(token)
            
        return (self.review_vocab_token, self.recommend_vocab)
        
        
