# Anomaly Detection with Conditioned Denoising Diffusion Models.

Official implementation of [DDAD](https://arxiv.org/abs/2305.15956) 


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anomaly-detection-with-conditioned-denoising/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=anomaly-detection-with-conditioned-denoising)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anomaly-detection-with-conditioned-denoising/anomaly-detection-on-visa)](https://paperswithcode.com/sota/anomaly-detection-on-visa?p=anomaly-detection-with-conditioned-denoising)


![Framework](images/DDAD_Framework.png)



## Requirements
This repository is implemented and tested on Python 3.8 and PyTorch 2.1.
To install requirements:

```setup
pip install -r requirements.txt
```

## Train and Evaluation of the Model
You can download the model checkpoints directly from [Checkpoints](https://drive.google.com/drive/u/0/folders/1FF83llo3a-mN5pJN8-_mw0hL5eZqe9fC) 

To train the denoising UNet, run:

```train
python main.py --train True
```

Modify the settings in the config.yaml file to train the model on different categories.


For fine-tuning the feature extractor, use the following command:

```domain_adaptation
python main.py --domain_adaptation True
```

To evaluate and test the model, run:

```detection
python main.py --detection True
```


## Dataset
You can download  [MVTec AD: MVTec Software](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) Benchmarks.
For preprocessing of VisA dataset check out the [Data preparation](https://github.com/amazon-science/spot-diff/tree/main) section of this repository.

The dataset should be placed in the 'datasets' folder. The training dataset should only contain one subcategory consisting of nominal samples, which should be named 'good'. The test dataset should include one category named 'good' for nominal samples, and any other subcategories of anomalous samples. It should be made as follows:

```shell
Name_of_Dataset
|-- Category
|-----|----- ground_truth
|-----|----- test
|-----|--------|------ good
|-----|--------|------ ...
|-----|--------|------ ...
|-----|----- train
|-----|--------|------ good
```




## Results
Running the code as explained in this file should achieve the following results for MVTec AD:

Anomaly Detection (Image AUROC) and Anomaly Localization (Pixel AUROC, PRO)

Expected results for MVTec AD:
| Category | Carpet | Grid |  Leather | Tile | Wood | Bottle |  Cable | Capsule | Hazel nut | Metalnut | Pill | Screw | Toothbrush | Transistor | Zipper |Average
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Detection | 99.3% | 100% | 100% | 100% | 100% | 100% | 99.4% | 99.4% | 100% | 100% | 100% | 99.0% | 100% | 100% | 100% | 99.8% 
| Localization | (98.7%,93.9%) |  (99.4%,97.3%) | (99.4%,97.7%) | (98.2%,93.1%) | (95.0%,82.9%) | (98.7%,91.8%) | (98.1%,88.9%) | (95.7%,93.4%) | (98.4%,86.7%) | (99.0%,91.1%) | (99.1%,95.5%) | (99.3%,96.3%) | (98.7%,92.6%) | (95.3%,90.1%) | (98.2%,93.2%) | (98,1%,92.3%)

The settings used for these results are detailed in the table.

| **Categories** | Carpet | Grid | Leather | Tile | Wood | Bottle | Cable | Capsule | Hazelnut | Metal nut | Pill | Screw | Toothbrush | Transistor | Zipper |
| -------------- | ------ | ---- | ------- | ---- | ---- | ------ | ----- | ------- | -------- | --------- | ---- | ----- | ----------- | ---------- | ------ |
| **\(w\)**       | 0      | 4    | 11      | 4    | 11   | 3      | 3     | 8       | 5        | 7         | 9    | 2     | 0           | 0          | 10     |
| **Training epochs** | 2500 | 2000 | 2000 | 1000 | 2000 | 1000 | 3000 | 1500 | 2000 | 3000 | 1000 | 2000 | 2000 | 2000 | 1000 |
| **FE epochs**   | 0      | 6    | 8       | 0    | 16   | 5      | 0     | 8       | 3        | 1         | 4    | 4     | 2           | 0          | 6      |


Following is the expected results on VisA Dataset. 

| Category | Candle | Capsules |  Cashew | Chewing gum | Fryum | Macaroni1 |  Macaroni2 | PCB1 | PCB2 | PCB3 | PCB4 | Pipe fryum | Average
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Detection | 99.9% | 100% | 94.5% | 98.1% | 99.0% | 99.2% | 99.2% | 100% |  99.7% | 97.2% | 100% | 100% | 98.9%
| Localization | (98.7%,96.6%) |  (99.5%,95.0%) | (97.4%,80.3%) | (96.5%,85.2%) | (96.9%,94.2%) | (98.7%,98.5%) | (98.2%,99.3%) | (93.4%,93.3%) | (97.4%,93.3%) | (96.3%,86.6%) | (98.5%,95.5%) | (99.5%,94.7%) |(97.6%,92.7%)

The settings used for these results are detailed in the table.

| **Categories**   | Candle | Capsules | Cashew | Chewing gum | Fryum | Macaroni1 | Macaroni2 | PCB1 | PCB2 | PCB3 | PCB4 | Pipe fryum |
| ---------------- | ------ | -------- | ------ | ------------ | ----- | --------- | --------- | ---- | ---- | ---- | ---- | ---------- |
| **\(w\)**         | 6      | 5        | 0      | 6            | 4     | 5         | 2         | 9    | 5    | 6    | 6    | 8          |
| **Training epochs** | 1000   | 1000     | 1750   | 1250         | 1000  | 500       | 500       | 500  | 500  | 500  | 500  | 500        |
| **FE epochs**     | 1      | 3        | 0      | 0            | 3     | 7         | 11        | 8    | 5    | 1    | 1    | 6          |


![Framework](images/Qualitative.png)

## Citation

```
@article{mousakhan2023anomaly,
  title={Anomaly Detection with Conditioned Denoising Diffusion Models},
  author={Mousakhan, Arian and Brox, Thomas and Tayyub, Jawad},
  journal={arXiv preprint arXiv:2305.15956},
  year={2023}
}
```

## Feedback

For any feedback or inquiries, please contact arian.mousakhan@gmail.com
