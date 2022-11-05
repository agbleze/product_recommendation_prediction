from Vocabulary import ReviewSequenceVocabulary, ReviewVocabulary
import numpy as np
import pandas as pd



class ReviewVectorizer:
    def __init__(self, review_df):
        self.review_df = review_df
        
        self.review_vocab = ReviewSequenceVocabulary()
        self.recommend_vocab = ReviewSequenceVocabulary()
        
    def vectorize(self, vector_length = -1):
        indices = [self.review_vocab.begin_seq_index]
        indices = indices.extend(self.review_vocab.add_token(token) for token in self.review_vocab._token_to_idx)
        indices = indices.append(self.review_vocab.mask_index)
        
        if vector_length >= 0:
            outer_vector = np.empty(len(indices), dtype=np.int64)
            outer_vector[:len(indices)] = indices
            outer_vector[len(indices):] = self.review_vocab.mask_index
            return outer_vector
        
    def get_vectorizer(self):
        return self.vectorize(vector_length=-1)
    
    
    def from_dataframe(self):
        for row, text_column in self.review_df.iterrows():
            for text in text_column['reviews.text'].split(" "):
                self.review_vocab_token = self.self.review_vocab.add_token(text) 
        
        for token in set(self.review_df['reviews.doRecommend'].split(" ")):
            self.recommend_vocab.add_token(token)
            
        return (self.review_vocab_token, self.recommend_vocab)
        
        
