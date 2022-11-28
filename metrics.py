import torch
from torchmetrics import ROC, AUROC, F1Score
import os


def metric(labels_list, predictions_max, predictions_mean, anomaly_map_list, GT_list, config):
    labels_list = torch.tensor(labels_list)
    predictions_max = torch.tensor(predictions_max)
    predictions_mean = torch.tensor(predictions_mean)

    resutls_embeddings = anomaly_map_list[0]
    for feature in anomaly_map_list[1:]:
        resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)

    GT_embeddings = GT_list[0]
    for feature in GT_list[1:]:
        GT_embeddings = torch.cat((GT_embeddings, feature), 0)

    resutls_embeddings = torch.tensor(resutls_embeddings)
    GT_embeddings = torch.tensor(GT_embeddings)

    roc = ROC()
    auroc = AUROC()
    fpr, tpr, thresholds = roc(predictions_max, labels_list)
    auroc_max = auroc(predictions_max, labels_list)
    auroc_mean = auroc(predictions_mean, labels_list)

    GT_embeddings = torch.flatten(GT_embeddings).type(torch.bool).cpu().detach()
    resutls_embeddings = torch.flatten(resutls_embeddings).cpu().detach()
    auroc_pixel = auroc(resutls_embeddings, GT_embeddings)

    f1 = F1Score()
    f1_scor_max = f1(predictions_max, labels_list)
    f1_scor_mean = f1(predictions_mean, labels_list)
    f1_score_pixel = f1(resutls_embeddings, GT_embeddings)

    if config.metrics.image_level_AUROC:
        print(f'AUROC_max: {auroc_max}   |   AUROC_mean: {auroc_mean}')
    if config.metrics.image_level_F1Score:
        print(f'F1SCORE_max: {f1_scor_max}    |  F1SCORE_mean: {f1_scor_mean}')
    if config.metrics.pixel_level_F1Score:
        print(f'f1_score_pixel: {f1_score_pixel}')
    if config.metrics.pixel_level_AUROC:
        print(f"auroc_pixel{auroc_pixel} ")

    with open('readme.txt', 'a') as f:
        f.write(
            f"AUROC_max: {auroc_max}   |   AUROC_mean: {auroc_mean}     |    auroc_pixel{auroc_pixel}    |     F1SCORE_max: {f1_scor_max}    |  F1SCORE_mean: {f1_scor_mean} |    f1_score_pixel: {f1_score_pixel}")
