import torch.nn as nn
import torch
import torch.nn.functional as F



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
            
        # 2. create CNN model
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size,
                      out_channels=num_channels, kernel_size=3
                      ),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2
                      ),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3
                      ),
            nn.ELU()
        )
        
        self._dropout_p = dropout_p
        
        # 3. create Multilayer perceptron
        self.fc1 = nn.Linear(in_channels=num_channels,
                             out_features=hidden_dim
                             )
        
        self.fc2 = nn.Linear(in_channels=hidden_dim, out_features=num_classes)
        
        
    def forward(self, x_in, apply_softmax=False):
        x_embedded = self.emb(x_in).permute(0, 2, 1)
        
        features = self.convnet(x_embedded)
        
        remaining_size = features.size(dim=2)
        features = F.avg_pool1d(input=features, kernel_size=remaining_size).squeeze(dim=2)
        features = F.dropout(input=features, p=self._dropout_p)
        
        intermediate_vector = F.relu(F.dropout(self.fc1(features), p=self._dropout_p))
        prediction_vector = self.fc2(intermediate_vector)
        
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)
            
        return prediction_vector
            
        





