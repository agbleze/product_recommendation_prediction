
from typing import dict


class ReviewVocabulary(object):
    """class for creating review vocabulary"""
    def __init__(self, token_to_idx: dict = None):
        if token_to_idx is None:
            self._token_to_idx = token_to_idx
            
        idx_to_token = {idx: token for token, idx in token_to_idx}
        
        
    def to_serializable(self):
        return {'token_to_idx': self._token_to_idx}
    
    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)
    
    
class ReviewSequenceVocabulary(ReviewVocabulary):
    def __init__(self,token_to_idx, unk_token: str = "<UNK>"):
        super(self, ReviewSequenceVocabulary).__init__(token_to_idx)
        
        
        

