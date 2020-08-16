import torch
import numpy as np
from tqdm import tqdm
from .metrics import metrics
from collections import defaultdict

def train(
        loader,
        model,
        optimizer,
        criterion,
        scheduler = None,
        masked_loss = False,
        show_progress = True
    ):
    model.train()
    losses = []
    
    if show_progress:
        loader = tqdm(loader)    
    
    for batch in loader:
        optimizer.zero_grad()
        dense_batch = batch.to_dense() # criterion may not support sparse tensor
        predictions = model(dense_batch)
        if masked_loss:
            mask = dense_batch > 0
            loss = criterion(predictions.masked_select(mask), dense_batch.masked_select(mask))
        else:
            loss = criterion(predictions, dense_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    if scheduler is not None:        
        scheduler.step()
    return losses
    

def evaluate(
        loader,
        eval_data,
        model,
        top_k = [10],
        report_coverage = True,
        show_progress = True
    ):
    model.eval()
    
    coverage_set = None
    if report_coverage:
        coverage_set = {k:set() for k in top_k}
        
    if show_progress:
        loader = tqdm(loader)        
    
    results = defaultdict(list)
    for i, batch in enumerate(loader):
        dense_batch = batch.to_dense().squeeze() # single user prediction
        with torch.no_grad():
            predictions = model(dense_batch)
        
        itemid = eval_data['items_data'][i]
        labels = eval_data['label_data'][i]
        predicted = predictions[itemid]
        
        for k in top_k:
            scores = metrics(predicted, labels, top_k=k, coverage_set=coverage_set[k])
            for score, metric in zip(scores, ['hr', 'arhr', 'ndcg']):
                results[f"{metric}@{k}"].append(score)
        
    results = {metric: np.mean(score) for metric, score in results.items()}
    
    if report_coverage:
        for k, cov in coverage_set.items():
            results[f"cov@{k}"] = len(cov)
    
    return results

def report_metrics(scores, epoch=None):
    log_str = f'Epoch: {epoch}' if epoch is not None else 'Scores'
    log = f"{log_str} | " + " | ".join(map(lambda x: f'{x[0]}: {x[1]:.6f}', scores.items()))
    print(log)