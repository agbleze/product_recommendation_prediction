
from typing import List,Dict
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd


class ReviewVocabulary(object):
    """class for creating review vocabulary"""
    def __init__(self, token_to_idx: Dict = None):
        if token_to_idx is None:
            self._token_to_idx = token_to_idx
            
        self._idx_to_token = {idx: token for token, idx in token_to_idx}
        
        
    def to_serializable(self):
        return {'token_to_idx': self._token_to_idx}
    
    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)
    
    def add_token(self, token):
        if token in self._token_to_idx:
            idx = self._token_to_idx[token]
        else:
            idx = len(self._token_to_idx)
            self._token_to_idx[token] = idx
            self._idx_to_token[idx] = token
        return idx
            
    def add_tokens(self, tokens: List):
        return [self.add_token(token) for token in tokens]
    
    
    def lookup_token(self, token: str) -> int:
        return self._token_to_idx[token]
    
    def lookup_index(self, index: int) -> str:
        if index in self._idx_to_token:
            return self._idx_to_token[index]
        else:
            raise KeyError(f"{index} is not in the Vocabulary")
        
    def __str__(self):
        return(f"ReviewVocabulary Size is {len(self)}")
    
    def __len__(self):
        return len(self._token_to_idx)
    
    
        
    
class ReviewSequenceVocabulary(ReviewVocabulary):
    def __init__(self, token_to_idx = None, unk_token: str = "<UNK>", mask_token: str = "<MASK>",
                 begin_seq_token: str = "<BEGIN>", end_seq_token: str = "<END>"):
        super(self, ReviewSequenceVocabulary).__init__(token_to_idx)
        
        self.mask_token = mask_token
        self.begin_seq_token = begin_seq_token
        self.end_seq_token = end_seq_token
        self.unk_token = unk_token
        
        self.mask_index = self.add_token(self.mask_token)
        self.begin_seq_index = self.add_token(self.begin_seq_token)
        self.end_seq_index = self.add_token(self.end_seq_token)
        self.unk_index = self.add_token(self.unk_token)
        
    def lookup_token(self, token: str) -> int:
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]
        
        
    def to_serializable(self):
        contents = super(self, ReviewSequenceVocabulary).to_serializable()
        contents = contents.update({'mask_token': self.mask_index,
                                    'unk_token': self.unk_token,
                                    'begin_seq_token': self.begin_seq_token,
                                    'end_seq_token': self.end_seq_token})
        
        
        
        
