import numpy as np
from consts import CONFIDENCE_THRESHOLD
from sklearn.metrics import average_precision_score, roc_auc_score

def mAP_and_mROCAUC(y_guess: np.ndarray, y_target: np.ndarray):
    y_true = (y_target > CONFIDENCE_THRESHOLD).astype(int)
    ap = average_precision_score(y_true, y_guess)
    mrocauc = roc_auc_score(y_true, y_guess, multi_class='ovr')

    return (ap, mrocauc)
'''
def mAP_and_mROCAUC(y_guess: np.ndarray, y_target: np.ndarray):
    # y_true and y_target are N * 26 matrices respectively (DIMENSIONS MUST MATCH and COVER WHOLE SET!)
    # it will return a 3-tuple of (count, mAP, mROCAUC) where count o

    num_classes = y_guess.shape[1]
    y_score = (y_target > CONFIDENCE_THRESHOLD).astype(int)

    y_total = np.stack([y_guess, y_score], axis=-1)
    # shape of y_total should be BATCH x 26 x 2

    for i in range(num_classes):
        class_column = precision_recall[:, i, :]
        class_column = np.sort(class_column, axis=1)

        y_guess_column = class_column[:, 0]
        y_score_column = class_column[:, 1]

        # confusion matrix
        true_positives = np.logical_and(y_score_column, y_guess_column) # AND, 1 1 -> 1
        false_positives = (y_guess_column - y_score_column == 1)
        true_negatives = ~np.logical_or(y_score_column, y_guess_column) # NOR 0 0 -> 1
        false_negatives = (y_guess_column - y_score_column == -1)
        # end confusion matrix

        # recall
'''