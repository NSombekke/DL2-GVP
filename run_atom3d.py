import argparse
import warnings
import random


warnings.filterwarnings('ignore')
"""
BiopythonDeprecationWarning: 'three_to_one' will be deprecated in a future release of Biopython in favor of 'Bio.PDB.Polypeptide.protein_letters_3to1'.
"""

parser = argparse.ArgumentParser()
parser.add_argument('task', metavar='TASK', choices=[
        'PSR', 'RSR', 'PPI', 'RES', 'MSP', 'SMP', 'LBA', 'LEP'
    ], help="{PSR, RSR, PPI, RES, MSP, SMP, LBA, LEP}")
parser.add_argument('--num-workers', metavar='N', type=int, default=4,
                   help='number of threads for loading data, default=4')
parser.add_argument('--smp-idx', metavar='IDX', type=int, default=0,
                   choices=list(range(20)),
                   help='label index for SMP, in range 0-19')
parser.add_argument('--lba-split', metavar='SPLIT', type=int, choices=[30, 60],
                    help='identity cutoff for LBA, 30 (default) or 60', default=30)
parser.add_argument('--batch', metavar='SIZE', type=int, default=8,
                    help='batch size, default=8')
parser.add_argument('--train-time', metavar='MINUTES', type=int, default=120,
                    help='maximum time between evaluations on valset, default=120 minutes')
parser.add_argument('--val-time', metavar='MINUTES', type=int, default=20,
                    help='maximum time per evaluation on valset, default=20 minutes')
parser.add_argument('--epochs', metavar='N', type=int, default=50,
                    help='training epochs, default=50')
parser.add_argument('--test', metavar='PATH', default=None,
                    help='evaluate a trained model')
parser.add_argument('--lr', metavar='RATE', default=1e-4, type=float,
                    help='learning rate')
parser.add_argument('--load', metavar='PATH', default=None, 
                    help='initialize first 2 GNN layers with pretrained weights')
parser.add_argument('--load_fullModel', metavar='PATH', default=None, 
                    help='load full model')
parser.add_argument('--seed', metavar='N', type=int, required=True,
                    help='random seed')
parser.add_argument('--transformer', action='store_true', default=False)
parser.add_argument('--bert_emb', action='store_true', default=False)
parser.add_argument('--bert_prediction', action='store_true', default=False)
parser.add_argument('--use_wandb', action='store_true', default=False)
args = parser.parse_args()

import gvp
from atom3d.datasets import LMDBDataset
import torch_geometric
from functools import partial
import gvp.atom3d
import torch.nn as nn
import tqdm, torch, time, os
import numpy as np
from atom3d.util import metrics
import sklearn.metrics as sk_metrics
from collections import defaultdict
import scipy.stats as stats
import wandb
print = partial(print, flush=True)

print("--------------------------")
models_dir = 'models'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)
model_id = float(time.time())

import os
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
warnings.filterwarnings("ignore", category=UserWarning)
def main():

    wandb.init(project="gvp", 
                mode="online" if args.use_wandb else "offline",
                config={         
                "Lisa": True,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "model_id": model_id,
                "batch_size": args.batch,
                "seed": args.seed,
                "transformer": args.transformer,
                "bert_emb": args.bert_emb,
                "bert_prediction":args.bert_prediction,
                "task": args.task,
                "device": device,
                "remark": "transformer + bert_emb + bert_prediction MLP",
            }
            )
    wandb.config.update(args)

    datasets = get_datasets(args.task, args.lba_split)
    dataloader = partial(torch_geometric.loader.DataLoader, 
                    num_workers=args.num_workers, batch_size=args.batch)
    if args.task not in ['PPI', 'RES']:
        dataloader = partial(dataloader, shuffle=True)
    trainset, valset, testset = map(dataloader, datasets)    
    model = get_model(args.task).to(device)

    # Magic
    wandb.watch(model, log_freq=100)

    print("-----------Info-----------\n\n")
    if args.bert_emb:
        print("doing protein bert")
    if args.test:
        print("--------testing--------")
        test(model, testset)
    else:
        if args.load:
            print("--------loading model--------")
            load(model, args.load)
        if args.load_fullModel:
            print("--------loading full model--------")
            model = get_model(args.task).to(device)
            model.load_state_dict(torch.load(args.load_fullModel))
        print("--------training model--------")
        train(model, trainset, valset)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()
        
def test(model, testset):
    model.load_state_dict(torch.load(args.test))
    model.eval()
    t = tqdm.tqdm(testset)
    metrics = get_metrics(args.task)
    targets, predicts, ids = [], [], []
    with torch.no_grad():
        for batch in t:
            pred = forward(model, batch, device)
            label = get_label(batch, args.task, args.smp_idx)
            if args.task == 'RES':
                pred = pred.argmax(dim=-1)
            if args.task in ['PSR', 'RSR']:
                ids.extend(batch.id)
            targets.extend(list(label.cpu().numpy()))
            predicts.extend(list(pred.cpu().numpy()))

    for name, func in metrics.items():
        if args.task in ['PSR', 'RSR']:
            func = partial(func, ids=ids)
        value = func(targets, predicts)
        print(f"{name}: {value}")

def train(model, trainset, valset):
                                
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_path, best_val = None, np.inf
    
    # Make model directory
    if args.transformer and not args.bert_emb:
        # Transformer normal has 8 heads
        root = f"{models_dir}/{args.task}/Transformer/{args.seed}/"
    elif args.transformer and args.bert_emb and not args.bert_prediction:
        root = f"{models_dir}/{args.task}/Transformer_bertEmb/{args.seed}/"
    elif args.transformer and args.bert_emb and args.bert_prediction:
        root = f"{models_dir}/{args.task}/GVP_Trans_bertPredic/{args.seed}/"
    else:
        root = f"{models_dir}/{args.task}/GVP/{args.seed}/"
    
    if not os.path.exists(root):
        os.makedirs(root)
    
    for epoch in range(args.epochs):    
        print(f"-----epoch {epoch}-----")
        # Model save path
        if args.transformer:
            path = os.path.join(root, f"{args.task}_{model_id}_{epoch}_TF.pt")
        elif args.bert_emb:
            path = os.path.join(root, f"{args.task}_{model_id}_{epoch}_PB.pt")
        else:
            path = os.path.join(root, f"{args.task}_{model_id}_{epoch}_GVP.pt")
        
        model.train()
        loss = loop(trainset, model, dataset_type="train", optimizer=optimizer, max_time=args.train_time)
        torch.save(model.state_dict(), path)
        print(f'\nEPOCH {epoch} TRAIN loss: {loss:.8f}')
        model.eval()
        with torch.no_grad():
            loss = loop(valset, model, dataset_type="validation", max_time=args.val_time)
        print(f'\nEPOCH {epoch} VAL loss: {loss:.8f}')
        if loss < best_val:
            best_path, best_val = path, loss
        print(f'BEST {best_path} VAL loss: {best_val:.8f}')

        wandb.log({"loss": loss})
    
def loop(dataset, model, dataset_type="train", optimizer=None, max_time=None):
    start = time.time()
    
    loss_fn = get_loss(args.task)
    t = tqdm.tqdm(dataset)
    total_loss, total_count = 0, 0
    
    log_interval = 10
    for batch_idx, (batch) in enumerate(t):

        if max_time and (time.time() - start) > 60*max_time: break
        if optimizer: optimizer.zero_grad()
        try:
            out = forward(model, batch, device)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise(e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue
            
        label = get_label(batch, args.task, args.smp_idx)
        loss_value = loss_fn(out, label)
        total_loss += float(loss_value)
        total_count += 1
        
        if optimizer:
            try:
                wandb.log({f"{dataset_type}_loss": loss_value})
                loss_value.backward()
                optimizer.step()
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e): raise(e)
                torch.cuda.empty_cache()
                print('Skipped batch due to OOM', flush=True)
                continue
      
        if batch_idx % log_interval == 0:
            wandb.log({f"{dataset_type}_batchidx_loss": loss_value})

           
        t.set_description(f"Avg loss: {total_loss/total_count:.8f} \t Total: {total_loss}, {total_count:.8f}")
        
    return total_loss / total_count

def load(model, path):
    params = torch.load(path)
    state_dict = model.state_dict()
    for name, p in params.items():
        if name in state_dict and \
               name[:8] in ['layers.0', 'layers.1'] and \
               state_dict[name].shape == p.shape:
            print("Loading", name)
            model.state_dict()[name].copy_(p)
        
#######################################################################

def get_label(batch, task, smp_idx=None):
    if type(batch) in [list, tuple]: batch = batch[0]
    if task == 'SMP':
        assert smp_idx is not None
        return batch.label[smp_idx::20]
    return batch.label

def get_metrics(task):
    def _correlation(metric, targets, predict, ids=None, glob=True):
        if glob: return metric(targets, predict)
        _targets, _predict = defaultdict(list), defaultdict(list)
        for _t, _p, _id in zip(targets, predict, ids):
            _targets[_id].append(_t)
            _predict[_id].append(_p)
        return np.mean([metric(_targets[_id], _predict[_id]) for _id in _targets])
        
    correlations = {
        'pearson': partial(_correlation, metrics.pearson),
        'kendall': partial(_correlation, metrics.kendall),
        'spearman': partial(_correlation, metrics.spearman)
    }
    mean_correlations = {f'mean {k}' : partial(v, glob=False) \
                            for k, v in correlations.items()}
    
    return {                       
        'RSR' : {**correlations, **mean_correlations},
        'PSR' : {**correlations, **mean_correlations},
        'PPI' : {'auroc': metrics.auroc},
        'RES' : {'accuracy': metrics.accuracy},
        'MSP' : {'auroc': metrics.auroc, 'auprc': metrics.auprc},
        'LEP' : {'auroc': metrics.auroc, 'auprc': metrics.auprc},
        'LBA' : {**correlations, 'rmse': partial(sk_metrics.mean_squared_error, squared=False)},
        'SMP' : {'mae': sk_metrics.mean_absolute_error}
    }[task]
            
def get_loss(task):
    if task in ['PSR', 'RSR', 'SMP', 'LBA']: return nn.MSELoss() # regression
    elif task in ['PPI', 'MSP', 'LEP']: return nn.BCELoss() # binary classification
    elif task in ['RES']: return nn.CrossEntropyLoss() # multiclass classification
    
def forward(model, batch, device):
    if type(batch) in [list, tuple]:
        batch = batch[0].to(device), batch[1].to(device)
    else:
        batch = batch.to(device)
    return model(batch)

def get_datasets(task, lba_split=30):
    data_path = {
        'RES' : 'data/atom3d-data/RES/raw/RES/data/',
                 
        'PPI' : 'atom3d-data/PPI/splits/DIPS-split/data/',
        'RSR' : 'atom3d-data/RSR/splits/candidates-split-by-time/data/',
        'PSR' : 'atom3d-data/PSR/splits/split-by-year/data/',
        'MSP' : 'data/atom3d-data/MSP/splits/split-by-sequence-identity-30/data/',
        'LEP' : 'atom3d-data/LEP/splits/split-by-protein/data/',
        'LBA' : f'atom3d-data/LBA/splits/split-by-sequence-identity-{lba_split}/data/',
        'SMP' : 'data/atom3d-data/SMP/splits/random/data/'
    }[task]
        
    if task == 'RES':
        # split_path = 'atom3d-data/RES/splits/split-by-cath-topology/indices/'
        split_path = 'data/atom3d-data/RES/raw/RES/data/indices/'
        # split_path = 'data/atom3d-data/RES/raw/RES/data/indices/'
        # split_path = 'atom3d-data/MSP/splits/split-by-sequence-identity-30/indices/'

        dataset = partial(gvp.atom3d.RESDataset, data_path)        
        trainset = dataset(split_path=split_path+'train_indices.txt')
        valset = dataset(split_path=split_path+'val_indices.txt')
        testset = dataset(split_path=split_path+'test_indices.txt')
    
    elif task == 'PPI':
        trainset = gvp.atom3d.PPIDataset(data_path+'train')
        valset = gvp.atom3d.PPIDataset(data_path+'val')
        testset = gvp.atom3d.PPIDataset(data_path+'test')
        
    else:
        transform = {                       
            'RSR' : gvp.atom3d.RSRTransform,
            'PSR' : gvp.atom3d.PSRTransform,
            'MSP' : gvp.atom3d.MSPTransform,
            'LEP' : gvp.atom3d.LEPTransform,
            'LBA' : gvp.atom3d.LBATransform,
            'SMP' : gvp.atom3d.SMPTransform,
        }[task]()
        
        trainset = LMDBDataset(data_path+'train', transform=transform)
        valset = LMDBDataset(data_path+'val', transform=transform)
        testset = LMDBDataset(data_path+'test', transform=transform)
        
    return trainset, valset, testset

def get_model(task):
    if args.transformer:
        print("Using TransformerConv")
    else:
        print("Using GVPConv")
    if args.bert_emb:
        print("Using ProteinBert embedding ")
    if args.bert_prediction:
        print("Using ProteinBert prediction ")
    return {
        'RES' : gvp.atom3d.RESModel(use_transformer=args.transformer, 
                                    use_bert_embedding=args.bert_emb, use_bert_predict=args.bert_prediction),
        'PPI' : gvp.atom3d.PPIModel(use_transformer=args.transformer),
        'RSR' : gvp.atom3d.RSRModel(use_transformer=args.transformer),
        'PSR' : gvp.atom3d.PSRModel(use_transformer=args.transformer),
        'MSP' : gvp.atom3d.MSPModel(use_transformer=args.transformer),
        'LEP' : gvp.atom3d.LEPModel(use_transformer=args.transformer),
        'LBA' : gvp.atom3d.LBAModel(use_transformer=args.transformer),
        'SMP' : gvp.atom3d.SMPModel(use_transformer=args.transformer)
    }[task]

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    main()
