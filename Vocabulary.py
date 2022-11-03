
from typing import dict


class ReviewVocabulary(object):
    """class for creating review vocabulary"""
    def __init__(self, token_to_idx: dict = None):
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
            
    
    
class ReviewSequenceVocabulary(ReviewVocabulary):
    def __init__(self, token_to_idx = None, unk_token: str = "<UNK>", mask_token: str = "<MASK>",
                 begin_seq_token: str = "<BEGIN>", end_seq_token: str = "<END>"):
        super(self, ReviewSequenceVocabulary).__init__(token_to_idx)
        
        
        

