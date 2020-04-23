# MixText
This repo contains codes for the following paper: 

*Jiaao Chen, Zichao Yang, Diyi Yang*: MixText: Linguistically-Informed Interpolation of Hidden Space for Semi-Supervised Text Classification. In Proceedings of the 58th Annual Meeting of the Association of Computational Linguistics (ACL'2020)

If you would like to refer to it, please cite the paper mentioned above. 


## Getting Started
These instructions will get you running the codes of MixText.

### Requirements
* Python 3.6 or higher
* Pytorch >= 1.3.0
* Pytorch_transformers (also known as transformers)
* Pandas, Numpy, Pickle
* Fairseq


### Code Structure
```
|__ data/
        |__ yahoo_answers_csv/ --> Datasets for Yahoo Answers
            |__ back_translate.ipynb --> Jupyter Notebook for back translating the dataset
            |__ classes.txt --> Classes for Yahoo Answers dataset
            |__ train.csv --> Original training dataset
            |__ test.csv --> Original testing dataset
            |__ de_1.pkl --> Back translated training dataset with German as middle language
            |__ ru_1.pkl --> Back translated training dataset with Russian as middle language

|__code/
        |__ transformers/ --> Codes copied form huggingface/transformers
        |__ read_data.py --> Codes for reading the dataset; forming labeled training set, unlabeled training set, development set and testing set; building dataloaders
        |__ normal_bert.py --> Codes for BERT baseline model
        |__ normal_train.py --> Codes for training BERT baseline model
        |__ mixtext.py --> Codes for our proposed MixText model
        |__ train.py --> Codes for training/testing MixText 
```

### Downloading the data
Please download the dataset and put them in the data folder. You can find Yahoo Answers, AG News, DB Pedia [here](https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset), IMDB [here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

### Pre-processing the data
We utilized [Fairseq](https://github.com/pytorch/fairseq) to perform back translation on the training dataset. Please refer to `./data/yahoo_answers_csv/back_translate.ipynb` for details.

Here, we have put two examples of back translated data, `de_1.pkl and ru_1.pkl`, in `./data/yahoo_answers_csv/` as well. You can directly use them for Yahoo Answers or generate your own back translated data followed the `./data/yahoo_answers_csv/back_translate.ipynb`.

### Training models
These section contains instructions for training models on Yahoo Answers using 10 labeled data per class for training.

#### Training BERT baseline model
Please run `./code/normal_train.py` to train the BERT baseline model (only use labeled training data):
```
python normal_train.py --gpu 0,1 --n-labeled 10 --data-path yahoo_answers_csv/ \
--batch-size 8 --epochs 20 --lrmain 0.000005 --lrlast 0.0005
```

#### Training TMix model
Please run `./code/train.py` to train the TMix model (only use labeled training data):
```
python train.py --gpu 0,1 --n-labeled 10 --data-path yahoo_answers_csv/ \
--batch-size 8 --batch-size-u 1 --epochs 50 --val-iteration 20 \
--lambda-u 0 --T 0.5 --alpha 2 --mix-layers-set 7 9 12 --seperate-mix True \
--lrmain 0.000005 --lrlast 0.0005
```


#### Training MixText model
Please run `./code/train.py` to train the MixText model (use both labeled and unlabeled training data):
```
python train.py --gpu 0,1,2,3 --n-labeled 10 \
--data-path yahoo_answers_csv/ --batch-size 4 --batch-size-u 8 --epochs 20 --val-iteration 1000 \
--lambda-u 1 --T 0.5 --alpha 16 --mix-layers-set 7 9 12 \
--lrmain 0.000005 --lrlast 0.0005
```


### Acknowledgements
We would like to thank the anonymous reviewers for their helpful comments, and Chao Zhang for his early feedback. We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan V GPU used for this research.Diyi is supported in part by a grant from Google.







