#%%
from helpers import (Namespace,
                     load_glove_from_file,
                     set_seed_everywhere,
                     handle_dirs,make_embedding_matrix,
                     make_train_state,update_train_state,
                     compute_accuracy, get_datapath
                     )

import ReviewVectorizer, ReviewClassifier
from ReviewDataset import ReviewDataset

review_dataset = ReviewDataset(dataset_path = get_datapath())

review_trainset, review_testset, review_validateset = review_dataset.split_set()


data, vectorizer = review_dataset.load_data_and_make_vectorizer()













# %%
