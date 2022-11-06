from torch.utils.data import dataloader, DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ReviewVectorizer
from typing import Dict, List, Optional

#review_Vec = ReviewVectorizer(review_df=review_df)

class ReviewDataset(Dataset):
    def __init__(self, dataset_path: str):
        
        self.dataset = pd.read_csv(dataset_path)
        self.review_vectorizer = ReviewVectorizer(review_df=self.dataset)
        
    def split_set(self, train_size: float = 0.7) -> Dict:
        """TODO:
        1. set seed with np.
        2. split data into train and evalute with 70:30 ratio
        3. split evaluate set into test and validate set with 50:50 ratio
        4. save train, test, validate set in a dictionary
        """
        np.random.seed(123)
        train_set, evaluate_set = train_test_split(self.dataset, train_size=0.7, random_state=123)
        
        validate_set, test_set = train_test_split(evaluate_set, test_size=0.5, random_state=123)
        
        return {'train_data': train_set,
                'test_data': test_set,
                'validate_data': validate_set
                }
        
    def lookup_datasplit(self, split='train_data'):
        self.split = split
        self.datasplit = self.split_set()[self.split]
        self.datasplit_size = len(self.datasplit)
        return (self.datasplit, self.datasplit_size)
           
    
    def load_data_and_make_vectorizer(self):
        self.train_df = self.split_set()['train_data']
        #self.review_vectorizer = ReviewVectorizer(review_df=self.dataset)
        review_vocab_token, recommend_vocab = self.review_vectorizer.from_dataframe()
        return (self.train_df, review_vocab_token)
    
    @property
    def load_data_and_load_vectorizer(self):
        self.vectorizer = self.review_vectorizer.get_vectorizer()
        return (self.train_df, self.vectorizer)
    
    @property
    def load_only_vectorizer(self):
        return self.vectorizer 
    
    
    def generate_batches(self, batch_size: int, drop_last=True, device='cpu'):
        dataloader = DataLoader(dataset=self.train_df, batch_size=batch_size,
                           shuffle=True, drop_last=drop_last)
        
        for data_dict in dataloader:
            output_data_dict = {}
            for name, tensor in data_dict.items():
                output_data_dict[name] = data_dict[name].to(device)
                
        return output_data_dict
         
    
    def get_num_batch(self, batch_size: int):
        len(self) // batch_size
        
    
    def __getitem__(self, index):
        self._train_df, self._train_df_size = self.lookup_datasplit()
        
        row = self._train_df.iloc[index]
        review_vector = self.review_vectorizer.vectorize(review_text=row['reviews.text'])
        
        recommend_index = self.review_vectorizer.recommend_vocab.lookup_token(row['reviews.doRecommend'])
        return {'x_data': review_vector, 'y_target': recommend_index}
    
    
    def __len__(self):
        return self._train_df_size
 
        
        
        
        
        
        
