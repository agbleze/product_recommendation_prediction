from torch.utils.data import dataloader, DataLoader, Dataset
import pandas as pd
import numpy as np
from sci


class ReviewDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        
        self.dataset = dataset
        
    def split_set(self):
        """TODO:
        1. set seed with np.
        2. split data into train and evalute with 70:30 ratio
        3. split evaluate set into test and validate set with 50:50 ratio
        4. save train, test, validate set in a dictionary
        """
        
    
    
    def load_data_and_make_vectorizer(self):
        pass
    
    def load_data_and_load_vectorizer(self):
        pass
    
    def load_only_vectorizer(self):
        pass
    
    
    def generate_batches(self, batch_size: int):
        pass
    
    def get_num_batch(self):
        pass
        
        
        
        
        
        #indices = [self.add_token(token)]
        
        
        
        
        







