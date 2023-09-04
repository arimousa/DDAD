# Anomaly Detection with Conditioned Denoising Diffusion Models.

This repository is the official implementation of DDAD

![Framework](images/DDAD_Framework.png)


## Requirements
This repository is implemented and tested on Python 3.8 and PyTorch 1.13.
To install requirements:

```setup
pip install -r requirements.txt
```

## Train and Evaluation of the Model
For tarining the denoising UNet, run:

```train
python main.py --train True
```

In the config.yaml file you can change the setting for trainnig. By chaning category, one can train the model on different categories.

For fine tuning the feature extractor, run:

```domain_adaptation
python main.py --domain_adaptation True
```
Note that in config.yaml, hyperparameter 'DA_epochs' determines the number of iteration for fine tuning. Depending on the number of trainin dataset and complexity of the category chagning this hyperparameter between 0 to 3 my help enormously.

For evaluation and testing the model, run:
```eval
python main.py --eval True
```
While we find 'w=4' a suitable number for reconstruction, increasing and decreasing this hyperparameter helps for better reconstruction.


Note that in config.yaml file 'epochs' referes to the number of training itarations. However, for evaluation the parameter 'load_chp' determines from which checkpoint the model should be loaded.'

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
We expect by running code as explained in this file achieve the following results. Nevertheless, slight changes may be expected due to different software and harware.
Following is the expected results on VisA Dataset. Anomaly Detection (Image AUROC) and Anomaly Localization (Pixel AUROC, PRO)
| Category | Candle | Capsules |  Cashew | Chewing gum | Fryum | Macaroni1 |  Macaroni2 | PCB1 | PCB2 | PCB3 | PCB4 | Pipe fryum | Average
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Detection | 99.9% | 97.9% | 98.4% | 99.0% | 98.8% | 100% | 99.2% | 99.9% |  99.2% | 100% | 99.9% | 99.8% | 99.3%
| Localization | (98.4%,95.2%) |  (99.6%,99.6%) | (93.0%,80.5%) | (97.6%,84.4%) | (93.6%,93.3%) | (99.3%,99.1%) | (99.2%,98.5%) | (94.3%,94.4%) | (97.0%,90.6%) | (98.1%,95.2%) | (98.3%,92.3%) | (96.5%,85.2%) |(97.0%,91.3%)

Expected results for MVTec AD:
| Category | Carpet | Grid |  Leather | Tile | Wood | Bottle |  Cable | Capsule | Hazel nut | Metalnut | Pill | Screw | Toothbrush | Transistor | Zipper |Average
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Detection | 97.8% | 100% | 100% | 100% | 99.8% | 100% | 99.7% | 98.1% | 99.9% | 99.0% | 98.6% | 99.5% | 100% | 99.6% | 100% | 99.5% 
| Localization | (98.0%,91.0%) |  (99.6%,98.5%) | (99.3%,98.3%) | (98.5%,96.7%) | (96.8%,90.0%) | (98.9%,94.8%) | (98.4%,90.9%) | (96.2%,90.7%) | (99.0%,87.3%) | (96.8%,91.8%) | (99.2%,95.6%) | (99.4%,92.0%) | (98.9%,95.0%) | (92.6%,87.2%) | (98.6%,94.1%) | (98,1%,92.9%)

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

Please reach out to arian.mousakhan@gmail.com
