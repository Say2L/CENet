import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def score_model(preds, labels, use_zero=False):
    mae = np.mean(np.absolute(preds - labels))
    corr = np.corrcoef(preds, labels)[0][1]
    non_zeros = np.array(
        [i for i, e in enumerate(labels) if e != 0 or use_zero])
    preds = preds[non_zeros]
    labels = labels[non_zeros]
    preds = preds >= 0
    labels = labels >= 0
    f_score = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)

    return acc, mae, corr, f_score