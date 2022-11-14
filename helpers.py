from argparse import Namespace
import os
import torch


args = Namespace(
    review_csv = "",
    vectorizer_filepath = "",
    model_state_file = "",
    save_dir = "",
    glove_filepath = "",
    use_glove=False,
    embedding_size=100,
    hidden_dim=100,
    num_channels=100,
    
    seed=1337,
    learning_rate=0.001,
    dropout_p=0.1,
    batch_size=128,
    num_epochs=5,
    early_stopping_criteria=5,
    
    cuda = True,
    catch_keyboard_interrupt=True,
    reload_from_files=False,
    expand_filepath_to_save_dir=True
)

def make_train_state(args):
    return {
        
    }



if not torch.cuda.isavailable:
    cuda = False


args.device = torch.device("cuda" if args.cuda else "cpu")
