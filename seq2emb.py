import gvp.atom3d

from transformers import T5Tokenizer, T5EncoderModel
import torch
import re

def get_sequence(trainset):
    for t in trainset:
        sequence = t.chain_sequences[0][-1]
        length = len(sequence)
        sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        return sequence, length

def get_embedding(sequence, length, model, tokenizer, device):
    ids = tokenizer.batch_encode_plus(sequence, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding_rpr = model(input_ids=input_ids,attention_mask=attention_mask)
    emb = embedding_rpr.last_hidden_state[0,:length] # shape (length x 1024)
    emb = emb.mean(dim=0) # shape (1024)
    return emb

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(device))

    transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print("Loading: {}".format(transformer_link))
    model = T5EncoderModel.from_pretrained(transformer_link)
    model.full() if device=='cpu' else model.half() # only cast to full-precision if no GPU is available
    model = model.to(device)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
    
    data_path = 'data/atom3d-data/RES/raw/RES/data/'
    split_path = 'data/atom3d-data/RES/splits/split-by-cath-topology/indices/'     
    trainset = gvp.atom3d.RESDataset(data_path, split_path=split_path+'train_indices.txt')
    sequence, length = get_sequence(trainset)
    
    embedding = get_embedding(sequence, length, model, tokenizer, device).detach().cpu()
    print(embedding)
    print(embedding.shape)