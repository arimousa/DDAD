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
from sklearn.metrics import roc_auc_score, roc_curve


class Metric:
    def __init__(self,labels_list, predictions, anomaly_map_list, gt_list, config) -> None:
        self.labels_list = labels_list
        self.predictions = predictions
        self.anomaly_map_list = anomaly_map_list
        self.gt_list = gt_list
        self.config = config
        self.threshold = 0.5
    
    def image_auroc(self):
        auroc_image = roc_auc_score(self.labels_list, self.predictions)
        return auroc_image
    
    def pixel_auroc(self):
        resutls_embeddings = self.anomaly_map_list[0]
        for feature in self.anomaly_map_list[1:]:
            resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
        resutls_embeddings =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 

        gt_embeddings = self.gt_list[0]
        for feature in self.gt_list[1:]:
            gt_embeddings = torch.cat((gt_embeddings, feature), 0)

        resutls_embeddings = resutls_embeddings.clone().detach().requires_grad_(False)
        gt_embeddings = gt_embeddings.clone().detach().requires_grad_(False)

        auroc_p = AUROC(task="binary")
        
        gt_embeddings = torch.flatten(gt_embeddings).type(torch.bool).cpu().detach()
        resutls_embeddings = torch.flatten(resutls_embeddings).cpu().detach()
        auroc_pixel = auroc_p(resutls_embeddings, gt_embeddings)
        return auroc_pixel
    
    def optimal_threshold(self):
        fpr, tpr, thresholds = roc_curve(self.labels_list, self.predictions)

        # Calculate Youden's J statistic for each threshold
        youden_j = tpr - fpr

        # Find the optimal threshold that maximizes Youden's J statistic
        optimal_threshold_index = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_threshold_index]
        self.threshold = optimal_threshold
        return optimal_threshold
    

    def pixel_pro(self):
        #https://github.com/hq-deng/RD4AD/blob/main/test.py#L337
        def _compute_pro(masks, amaps, num_th = 200):
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
            binary_amaps = np.zeros_like(amaps)
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
        
        pro = _compute_pro(self.gt_list, self.anomaly_map_list, num_th = 200)
        return pro
    

    def miscalssified(self):
        predictions = torch.tensor(self.predictions)
        labels_list = torch.tensor(self.labels_list)
        predictions0_1 = (predictions > self.threshold).int()
        for i,(l,p) in enumerate(zip(labels_list, predictions0_1)):
            print('Sample : ', i, ' predicted as: ',p.item() ,' label is: ',l.item(),'\n' ) if l != p else None

