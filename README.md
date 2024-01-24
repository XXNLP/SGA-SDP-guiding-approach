[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/downstream-model-design-of-pre-trained/relation-extraction-on-nyt)](https://paperswithcode.com/sota/relation-extraction-on-nyt?p=downstream-model-design-of-pre-trained)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/downstream-model-design-of-pre-trained/relation-extraction-on-semeval-2010-task-8)](https://paperswithcode.com/sota/relation-extraction-on-semeval-2010-task-8?p=downstream-model-design-of-pre-trained)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/downstream-model-design-of-pre-trained/relation-extraction-on-webnlg)](https://paperswithcode.com/sota/relation-extraction-on-webnlg?p=downstream-model-design-of-pre-trained)
# SGA

This is a model called SGA for Relation Extraction, supporting paper [*Enhancing Relation Extraction using Multi-Task Learning with SDP evidence*.](https://arxiv.org/)

Part of this code are revised based on [OPENNRE](https://github.com/thunlp/OpenNRE).

## Datasets

You can get Datasets from [OPENNRE](https://github.com/thunlp/OpenNRE) and [WebNLG 2.0](https://gitlab.com/shimorina/webnlg-dataset/tree/master/release_v2)

## Getting Start
 
Set your own paths in example/configs.py, including pre-trained model path, root path of data and output name.
Run example/sga_trainer.py with args **dataset** and **mode**. **dataset** can be nyt20, semeval or webnlg.

* Prepration:  first, install Spacy to for parsing all data to get SDP for each entity pair. More detail about Spacy reference https://github.com/explosion/spacy-models

 ```
python preprocess.py
python parse.py semeval
```

* Train&Evaluation: **mode** can be t for training and e for evaluation. For example ,to train SemEval, try
 ```
python sga_trainer semeval t
```

## Logs

If you are not able to run these codes, you can also check all the logs in ./logs.

**This code is not the final version, we will update it soon.**