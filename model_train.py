#%%
from helpers import (args,
                     load_glove_from_file,
                     set_seed_everywhere,
                     handle_dirs,make_embedding_matrix,
                     make_train_state,update_train_state,
                     compute_accuracy, get_datapath
                     )

from ReviewClassifier import Classifier
from ReviewVectorizer import ReviewVectorizer
from ReviewDataset import ReviewDataset

import torch, os

#%%
if args.expand_filepaths_to_save_dir:
    args.vectorizer_file = os.path.join(args.save_dir,
                                        args.vectorizer_file)

    args.model_state_file = os.path.join(args.save_dir,
                                         args.model_state_file)
    
    print("Expanded filepaths: ")
    print("\t{}".format(args.vectorizer_file))
    print("\t{}".format(args.model_state_file))
    
#%% Check CUDA
if not torch.cuda.is_available():
    args.cuda = False
    
args.device = torch.device("cuda" if args.cuda else "cpu")
print("Using CUDA: {}".format(args.cuda))

# Set seed for reproducibility
set_seed_everywhere(args.seed, args.cuda)

# handle dirs
handle_dirs(args.save_dir)

#%%
dataset = ReviewDataset(dataset_path = get_datapath())
if args.reload_from_files:
    #dataset = ReviewDataset(dataset_path = get_datapath())
    dataset = dataset.load_data_and_load_vectorizer
else:
    dataset = dataset.load_data_and_make_vectorizer
    dataset.save_vectorizer(args.vectorizer_file)
vectorizer = dataset.get_vectorizer()
    

#%%
review_dataset = ReviewDataset(dataset_path = get_datapath())

#%%
review_trainset, review_testset, review_validateset = review_dataset.split_set()


data, vectorizer = review_dataset.load_data_and_make_vectorizer()













# %%
