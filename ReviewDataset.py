from torch.utils.data import dataloader, DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ReviewVectorizer

#review_Vec = ReviewVectorizer(review_df=review_df)

class ReviewDataset(Dataset):
    def __init__(self, dataset_path: str):
        
        self.dataset = pd.read_csv(dataset_path)
        
    def split_set(self, train_size: float = 0.7):
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
           
    
    def load_data_and_make_vectorizer(self):
        self.train_df = self.split_set()['train_data']
        review_vectorizer = ReviewVectorizer()
    
    def load_data_and_load_vectorizer(self):
        pass
    
    def load_only_vectorizer(self):
        pass
    
    
    def generate_batches(self, batch_size: int):
        pass
    
    def get_num_batch(self):
        pass
        
        
        
        
        
        #indices = [self.add_token(token)]
        
        
        
        
        







