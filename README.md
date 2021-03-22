# News deduplication model

Semi-supervised machine learning model trained with objective to detect news stories describing same events. 
It is largely inspired by ["Neural Duplicate Question Detection without Labeled Training Data (2019)"](https://www.aclweb.org/anthology/D19-1171/) paper.

This model uses ideas from "weak supervision with title-body pairs (WS-TB)" method described in the paper.
It assumes that first part of news body text rephrases news title and therefore a pair `(title(article), body(article))` 
could be used as a positive sample in model training. Additionally, same number of `(body(article), title(article))`
pairs were added to the training set to increase its robustness and to rectify biases introduced by different distribution
of `title(article)` and `body(article)` features. 

Negative samples are generated from `(title(article(i)), body(article(j)))`, where `article(i)` and `article(j)` are 
articles that belong to the same topic, but were published with at least 1 day difference in time. 
See `gen_news_duplicates_data` in `train.ipynb` for more details. 

## Using the model

Load the model
```
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch 
model_news_dup = ClassificationModel(
    'roberta',
    'model/roberta_news_duplicates',
    use_cuda=torch.cuda.is_available(), 
    args=ClassificationArgs(
        fp16=True, 
        dataloader_num_workers=1,        
        use_multiprocessing_for_evaluation=False,
        eval_batch_size=32,
    )    
) 
```
Run prediction.

Negative example:
```
model_news_dup.predict([
    [
        'Coronavirus: Third wave will "wash up on our shores", warns Johnson', 
        'Some 18,000 people have been evacuated from severe floods across New South Wales (NSW) in Australia, with more heavy rainfall predicted.'
    ]
])[0]
```
```
> array([0], dtype=int64)
```

Positive example:

```
model_news_dup.predict([
    [
        'Coronavirus: Third wave will "wash up on our shores", warns Johnson', 
        'Boris Johnson has warned the effects of a third wave of coronavirus will "wash up on our shores" from Europe. The PM said the UK should be "under no illusion" we will "feel effects" of growing cases on the continent'
    ]
])[0]
```
```
array([1], dtype=int64)
```

## Training

Use `train.ipynb` to retrain the model.
The model is a fine-tuned RoBERTa. 

Training data is stored in `data` folder. 
It consists of ~113000 news articles over the period of 2019-2020.


