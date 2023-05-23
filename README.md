# Anomaly Detection with Conditioned Denoising Diffusion Models.

This repository is the official implementation of DDAD

![Framework](imges/DDAD_Framework.png)


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Train and Evaluation of the Model
For tarining the denoising UNet, run:

```train
python --train True
```

For fine tuning the feature extractor, run:

```domain_adaptation
python --domain_adaptation True
```
Note that in config.yaml, hyperparameter 'DA_epochs' determines the number of iteration for fine tuning. Depending on the number of trainin dataset and complexity of the category chagning this hyperparameter between 0 to 3 my help enormously.

For evaluation and testing the model, run:
```eval
python --eval True
```
While we find 'w=4' a suitable number for reconstruction, increasing and decreasing this hyperparameter helps for better reconstruction.


Note that in config.yaml file 'epochs' referes to the number of training itarations. However, for evaluation the parameter 'load_chp' determines from which checkpoint the model should be loaded.'

## Dataset
You can download  [MVTec AD: MVTec Software](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) Benchmarks.
For preprocessing of VisA dataset check out the [Data preparation](https://github.com/amazon-science/spot-diff/tree/main) section of this repository.


## Feedback
You can reach out to arian.mousakhan@gmail.com for questions or suggestions.
