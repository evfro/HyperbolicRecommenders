import torch
import numpy as np
from tqdm import tqdm
from .metrics import metrics
from collections import defaultdict


def train(loader, model, optimizer, criterion, scheduler=None, show_progress=True):
    model.train()
    losses = []
    
    if show_progress:
        loader = tqdm(loader)
    
    for users, items, labels in loader:
        optimizer.zero_grad()
        loss = criterion(model(users, items), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    if scheduler is not None:        
        scheduler.step()
    return losses


def validate(loader, model, top_k=[10], report_coverage=True, show_progress=True):
    model.eval()
    data = defaultdict(list)
    
    coverage_set = None
    if report_coverage:
        coverage_set = set()

    if show_progress:
        loader = tqdm(loader)
    
    for users, items, labels in loader:
        with torch.no_grad():
            predictions = model(users, items)
            for k in top_k:
                hits, arhr, dcgs = metrics(predictions, labels, top_k=k, coverage_set=coverage_set)
                data[f"hr@{k}"].append(hits)
                data[f"arhr@{k}"].append(arhr)
                data[f"ndcg@{k}"].append(dcgs)
    
    output = {}
    for metric in ["hr", "arhr", "ndcg"]:
        for k in top_k:
            name = f"{metric}@{k}"
            output[name] = np.mean(data[name])
            
    if report_coverage:
        for k in top_k:
            name = f"cov@{k}"
            output[name] = len(coverage_set)
        
    return output