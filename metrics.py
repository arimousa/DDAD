import torch
from torchmetrics import ROC, AUROC, F1Score
import os
from torchvision.transforms import transforms
from skimage import measure
import pandas as pd
from statistics import mean
import numpy as np
from sklearn.metrics import auc
from sklearn import metrics

#https://github.com/hq-deng/RD4AD/blob/main/test.py#L337
def compute_pro(masks, amaps, num_th = 200):
    resutls_embeddings = amaps[0]
    for feature in amaps[1:]:
        resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
    amaps =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 
    amaps = amaps.squeeze(1)
    amaps = amaps.cpu().detach().numpy()
    gt_embeddings = masks[0]
    for feature in masks[1:]:
        gt_embeddings = torch.cat((gt_embeddings, feature), 0)
    masks = gt_embeddings.squeeze(1).cpu().detach().numpy()
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks , binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        # print(f"Threshold: {th}, FPR: {fpr}, PRO: {mean(pros)}")

        df = pd.concat([df, pd.DataFrame({"pro": mean(pros), "fpr": fpr, "threshold": th}, index=[0])], ignore_index=True)
        # df = df.concat({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def metric(labels_list, predictions, anomaly_map_list, gt_list, config):
    labels_list = torch.tensor(labels_list)
    predictions = torch.tensor(predictions)
    pro = compute_pro(gt_list, anomaly_map_list, num_th = 200)
    resutls_embeddings = anomaly_map_list[0]
    for feature in anomaly_map_list[1:]:
        resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
    resutls_embeddings =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 

    gt_embeddings = gt_list[0]
    for feature in gt_list[1:]:
        gt_embeddings = torch.cat((gt_embeddings, feature), 0)

    resutls_embeddings = resutls_embeddings.clone().detach().requires_grad_(False)
    gt_embeddings = gt_embeddings.clone().detach().requires_grad_(False)

    auroc = AUROC(task="binary")
    
    auroc_score = auroc(predictions, labels_list)

    gt_embeddings = torch.flatten(gt_embeddings).type(torch.bool).cpu().detach()
    resutls_embeddings = torch.flatten(resutls_embeddings).cpu().detach()

    auroc_pixel = auroc(resutls_embeddings, gt_embeddings)

    r_gt_embeddings = gt_embeddings.cpu().detach().numpy().ravel()
    r_resutls_embeddings = resutls_embeddings.cpu().detach().numpy().ravel()
    precision, recall, thresholds = metrics.precision_recall_curve(
        r_gt_embeddings.astype(int), r_resutls_embeddings
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    thresholdOpt = thresholds[np.argmax(F1_scores)]


    predictions0_1 = (predictions > thresholdOpt).int()
    for i,(l,p) in enumerate(zip(labels_list, predictions0_1)):
        print('Sample : ', i, ' predicted as: ',p.item() ,' label is: ',l.item(),'\n' ) if l != p else None


    if config.metrics.image_level_AUROC:
        print(f'AUROC: {auroc_score}')
    if config.metrics.pixel_level_AUROC:
        print(f"AUROC pixel level: {auroc_pixel} ")
    if config.metrics.pro:
        print(f'PRO: {pro}')
        
    auroc = auroc.reset()
    return thresholdOpt
