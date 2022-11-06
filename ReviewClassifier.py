import torch.nn as nn
import torch



class Classifier(nn.Module):
    def __init__(self, embedding_size, num_embedding, num_channels,
                 hidden_dim, num_classes, dropout_p, pretrained_embeddings=None,
                 padding_idx=0) -> None:
        super().__init__()
        
       # 1. create embeddings
        if pretrained_embeddings is None:
            self.emb = nn.Embedding(num_embeddings=num_embedding,
                                    embedding_dim=embedding_size,
                                    padding_idx=padding_idx
                                    )
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim=embedding_size, 
                                    num_embeddings=num_embedding,
                                    padding_idx=padding_idx,
                                    _weight=pretrained_embeddings
                                    )
            
        # 2. create model
        self.convnet = nn.Sequential(
            
        )
            
        





