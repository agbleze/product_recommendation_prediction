from argparse import Namespace
import os
import torch
import numpy as np


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
        "stop_early": False,
        "early_stopping_step": 0,
        "early_stopping_best_val": 1e8,
        "learning_rate": args.learning_rate,
        "epoch_index": 0,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": -1,
        "test_acc": -1,
        "model_filename": args.model_state_file      
    }
    
def update_train_state(args, model, train_state: dict):
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filenae'])
        train_state['stop_early'] = False
        
    # save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]
        
        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            train_state["early_stopping_step"] += 1
            
        # loss decreased
        else:
            # save the best model
            if loss_t < train_state["early_stopping_best_val"]:
                torch.save(model.state_dict(), train_state["model_filename"])
                
            # reset early stopping step
            train_state['early_stopping_step'] = 0
            
        # stop early?
        train_state['stop_early'] =\
            train_state['early_stopping_step'] >= args.early_stopping_criteria
            
    return train_state


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all()
        

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
        
        
def load_glove_from_file(glove_filepath):
    word_to_index = {}
    embeddings = []
    
    with open(glove_filepath) as fp:
        for index, line in enumerate(fp):
            line = line.split(" ") # each line word num1 num2 ...
            word_to_index[line[0]] = index
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)
    

def make_embedding_matrix(glove_filepath, words):
    """create embedding matrix from specific set of words

    Args:
        glove_filepath (str): filepath to the glove embeddings
        words (list): list of words in the dataset
    """
    word_to_idx, glove_embedding = load_glove_from_file(glove_filepath)
    embedding_size = glove_embedding.shape[1]
    
    final_embeddings = np.zeros((len(words), embedding_size))
    
    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embedding[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i
            
    return final_embeddings

if not torch.cuda.isavailable:
    cuda = False


args.device = torch.device("cuda" if args.cuda else "cpu")
