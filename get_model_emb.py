import torch
import re
import numpy as np
import gc

import re

model_name = "prot_bert"
# model_name = "prot_t5_xl_half_uniref50-enc"

if model_name == "prot_bert":
    from transformers import BertModel, BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(f"Rostlab/{model_name}", do_lower_case=False )
    model = BertModel.from_pretrained(f"Rostlab/{model_name}")
if model_name == "prot_t5_xl_half_uniref50-enc":
    from transformers import T5EncoderModel, T5Tokenizer
    # tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    # model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")


gc.collect()
NUM_ATOM_TYPES = 9
_SEQ_EMBED_SIZE = 1024
_element_mapping = lambda x: {
    'H' : 0,
    'C' : 1,
    'N' : 2,
    'O' : 3,
    'F' : 4,
    'S' : 5,
    'Cl': 6, 'CL': 6,
    'P' : 7
}.get(x, 8)

_amino_acids = lambda x: {
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLU': 5,
    'GLN': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}.get(x, 20)

map_amino_3to1 = lambda x: {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
}.get(x, '')



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device = 'cpu'
torch.cuda.empty_cache()
torch.max_split_size_mb = 256

model = model.to(device)
model = model.eval()

amino_acids_1_letter = [
    'A',  # Alanine
    'R',  # Arginine
    'N',  # Asparagine
    'D',  # Aspartic Acid
    'C',  # Cysteine
    'Q',  # Glutamine
    'E',  # Glutamic Acid
    'G',  # Glycine
    'H',  # Histidine
    'I',  # Isoleucine
    'L',  # Leucine
    'K',  # Lysine
    'M',  # Methionine
    'F',  # Phenylalanine
    'P',  # Proline
    'S',  # Serine
    'T',  # Threonine
    'W',  # Tryptophan
    'Y',  # Tyrosine
    'V',  # Valine
    # '[MASK]',  # Mask
]

sequences_Example = [" ".join(amino_acids_1_letter)]
print(sequences_Example)
sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)

input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

with torch.no_grad():
    embedding = model(input_ids=input_ids,attention_mask=attention_mask)

embedding = embedding.last_hidden_state.cpu().numpy()

features = [] 
for seq_num in range(len(embedding)):
    seq_len = (attention_mask[seq_num] == 1).sum()
    seq_emd = embedding[seq_num][:seq_len-1]
    features.append(seq_emd)

ATOM_TYPES_EMB = features[0]

# [20,1024]
print(ATOM_TYPES_EMB.shape)

torch.save(ATOM_TYPES_EMB, f'data/AMINO_TYPES_andMask_EMB_{model_name}.pt')