import torch
from torchmetrics import ROC, AUROC, F1Score
import os
from torchvision.transforms import transforms


def metric(labels_list, predictions, anomaly_map_list, GT_list, config):
    labels_list = torch.tensor(labels_list)
    predictions = torch.tensor(predictions)
    

    resutls_embeddings = anomaly_map_list[0]
    for feature in anomaly_map_list[1:]:
        resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
    resutls_embeddings =  ((resutls_embeddings - resutls_embeddings.min())/ (resutls_embeddings.max() - resutls_embeddings.min())) 

    GT_embeddings = GT_list[0]
    for feature in GT_list[1:]:
        GT_embeddings = torch.cat((GT_embeddings, feature), 0)

    resutls_embeddings = resutls_embeddings.clone().detach().requires_grad_(False)
    GT_embeddings = GT_embeddings.clone().detach().requires_grad_(False)

    roc = ROC(task="binary")
    auroc = AUROC(task="binary")

    fpr, tpr, thresholds = roc(predictions, labels_list)
    auroc_score = auroc(predictions, labels_list)

    GT_embeddings = torch.flatten(GT_embeddings).type(torch.bool).cpu().detach()
    resutls_embeddings = torch.flatten(resutls_embeddings).cpu().detach()

    auroc_pixel = auroc(resutls_embeddings, GT_embeddings)
    thresholdOpt_index = torch.argmax(tpr - fpr)
    thresholdOpt = thresholds[thresholdOpt_index]

    f1 = F1Score(task="binary")
    predictions0_1 = (predictions > thresholdOpt).int()
    for i,(l,p) in enumerate(zip(labels_list, predictions0_1)):
        print('sample : ', i, ' prediction is: ',p.item() ,' label is: ',l.item() , 'prediction is : ', predictions[i].item() ,'\n' ) if l != p else None

    f1_scor = f1(predictions0_1, labels_list)
    f1_pixel = f1(resutls_embeddings, GT_embeddings)

    if config.metrics.image_level_AUROC:
        print(f'AUROC: {auroc_score}')
    if config.metrics.pixel_level_AUROC:
        print(f"AUROC pixel level: {auroc_pixel} ")
    if config.metrics.image_level_F1Score:
        print(f'F1SCORE: {f1_scor}')
    if config.metrics.pixel_level_F1Score:
        print(f'F1SCORE pixel level: {f1_pixel}')

    with open('readme.txt', 'a') as f:
        f.write(
            f"{config.data.category} \n")
        f.write(
            f"AUROC: {auroc_score}       |    auroc_pixel: {auroc_pixel}    |     F1SCORE: {f1_scor}    |     F1SCORE_pixel: {f1_pixel}   \n")
    roc = roc.reset()
    auroc = auroc.reset()
    f1 = f1.reset()
    return thresholdOpt
