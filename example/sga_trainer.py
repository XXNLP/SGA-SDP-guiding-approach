# coding:utf-8
import torch
import json
import sys
from torch.utils.data import DataLoader
import pickle
import datetime


sys.path.append("../sga-master")
sys.path.append("../sga-master/example")

CUDA_LAUNCH_BLOCKING=0
#device = torch.device("cpu")pyth

from opennre import encoder, model, framework
from opennre.framework.data_loader import SentenceREDataset
from opennre.framework.f1_metric import F1Metric
from example import configs as cf
from opennre.model.RE_loss import RELoss
from opennre.model.RE_loss_softmax import RELossSoftmax
import os


def train(dataset_name, batch_size=50, num_workers=4, max_epoch=15, lr=3e-5, weight_decay=1e-5, add_subject_loss=False,
          eval=False, continue_train=False, large_bert=False, subject_1=False, use_cls=True, softmax=False,
          opt='adam', seed=31415926535897932, cuda_device=0, sort=True, metric="micro_f1", model_path=None, sdp_loss_lambda=0.16
          ):
    print("@@@@@@@@@@@ args @@@@@@@@@@@")
    print(locals())
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ["CUDA_LAUNCH_BLOCKING"] = '0'

    if seed is not None:
        torch.manual_seed(seed)

    root_path = cf.rootpath
    dataset_file = ["train.json", "dev.json", "test.json"]
    dataset_pkl_file = ["train.pkl", "dev.pkl", "test.pkl"]

    if large_bert:
        bert_path, bert_large = cf.modelpath, True
    else:
        bert_path, bert_large = cf.modelpath, False

    ckpt = './ckpt/%s_%s_%s_%s_%s_%s_%s_%s_bert.th' % (
        cf.outputname,
        dataset_name,
        "softmax" if softmax else "sigmoid",
        "withCLS" if use_cls else "withoutCLS",
        "-1" if subject_1 else "-2",
        "1024" if bert_large else "768",
        "lambda"+str(sdp_loss_lambda),
        datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S")) if model_path is None else model_path


    def get_dataset(_model):
        if all(map(lambda x: os.path.exists(os.path.join(root_path, dataset_name, x)), dataset_pkl_file)):
            dataset = list(
                map(lambda x: pickle.load(open(os.path.join(root_path, dataset_name, x), "rb")), dataset_pkl_file))

            if softmax:
                list(map(lambda x: x.split(), dataset))
        else:
            dataset = list(
                map(lambda x: SentenceREDataset(path=os.path.join(root_path, dataset_name, x), rel2id=rel2id,
                                                tokenizer=_model.sentence_encoder.tokenize, kwargs=None,
                                                sort=sort),
                    dataset_file))

            list(map(lambda x, y: pickle.dump(x, open(os.path.join(root_path, dataset_name, y), "wb")), dataset,
                     dataset_pkl_file))

        if dataset_name in ["nyt10", "nyt10_1", "nyt10_2"]:
            list(map(lambda x: x.set_max_words(100), dataset))
            list(map(lambda x: x.remove_na(), dataset))
            # list(map(lambda x: x.remove_repeat(), dataset))
            list(map(lambda x: x.char_idx_to_word_idx(), dataset))
            for d in dataset:
                d.NA_id = -1
        if dataset_name in ["semeval_1"]:
            for d in dataset:
                d.NA_id = -1
        #if dataset_name in ["kbp37"]:
        #    for d in dataset:
        #        d.NA_id = -1

        if dataset_name in ["webnlg", "webnlg_1"]:
            for d in dataset:
                d.NA_id = -1


        dataset_loader = list(map(
            lambda x: DataLoader(dataset=x, batch_size=batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers, collate_fn=SentenceREDataset.collate_fn), dataset))

        return dataset_loader

    rel2id = json.load(open(os.path.join(root_path, dataset_name, 'rel2id.json')))

    sentence_encoder = encoder.BERTHiddenStateEncoder(pretrain_path=bert_path)
    _model = model.SGA(sentence_encoder, len(rel2id), rel2id, num_token_labels=2, subject_1=subject_1, use_cls=use_cls)

    train_loader, val_loader, test_loader = get_dataset(_model)

    _framework = framework.SentenceRE(
        train_loader=train_loader,
        val_loader=val_loader if dataset_name not in ["nyt10", "nyt10_1"] else test_loader,
        test_loader=test_loader,
        model=_model,
        ckpt=ckpt,
        max_epoch=max_epoch,
        lr=lr,
        weight_decay=weight_decay,
        opt=opt,
        add_subject_loss=add_subject_loss,
        loss_func=RELossSoftmax() if softmax else RELoss(),
        metric=F1Metric(multi_label=not softmax,
                        na_id=train_loader.dataset.NA_id,
                        ignore_na=dataset_name == "semeval",
                        rel2id=rel2id,
                        print_error_prob=1
                        ),
        sdp_loss_lambda=sdp_loss_lambda,
    )

    if not eval:
        if continue_train:
            _framework.parallel_model.load_state_dict(torch.load(ckpt).state_dict())
        _framework.train_model(metric=metric)
    _framework.parallel_model.load_state_dict(torch.load(ckpt).state_dict())

    # print("TRAIN---------------------------")
    # result = _framework.eval_model(_framework.train_loader)
    # print('Accuracy on test set: {}'.format(result['acc']))
    # print('Micro Precision: {}'.format(result['micro_p']))
    # print('Micro Recall: {}'.format(result['micro_r']))
    # print('Micro F1: {}'.format(result['micro_f1']))
    #
    # print("DEV---------------------------")
    # result = _framework.eval_model(_framework.val_loader)
    # print('Accuracy on test set: {}'.format(result['acc']))
    # print('Micro Precision: {}'.format(result['micro_p']))
    # print('Micro Recall: {}'.format(result['micro_r']))
    # print('Micro F1: {}'.format(result['micro_f1']))

    print("TEST---------------------------")
    result = _framework.eval_model(_framework.test_loader)
    print('Accuracy on test set: {}'.format(result['acc']))
    print('Micro Precision: {}'.format(result['micro_p']))
    print('Micro Recall: {}'.format(result['micro_r']))
    print('Micro F1: {}'.format(result['micro_f1']))

    if os.path.exists(os.path.join(root_path, dataset_name, "test_sample.json")):
        test_sample_dataset = SentenceREDataset(path=os.path.join(root_path, dataset_name, "test_sample.json"), rel2id=rel2id,
                                                tokenizer=_model.sentence_encoder.tokenize, kwargs=None,
                                                sort=sort)
        test_sample_loader = DataLoader(dataset=test_sample_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                        num_workers=num_workers, collate_fn=SentenceREDataset.collate_fn)
        print("TEST-Sample--------------------")
        result = _framework.eval_model(test_sample_loader)
        print('Accuracy on test set: {}'.format(result['acc']))
        print('Micro Precision: {}'.format(result['micro_p']))
        print('Micro Recall: {}'.format(result['micro_r']))
        print('Micro F1: {}'.format(result['micro_f1']))
        _framework.metric.df.to_excel(os.path.join(root_path, dataset_name, "res.xlsx"))


def get_ablation_args(dataset, max_epoch, batch_size, **kwargs):
    _args_list = []

    args = {"dataset_name": dataset, "max_epoch": max_epoch, "batch_size": batch_size,
            "subject_1": False, "use_cls": True, "softmax": False, }
    args.update(kwargs)
    _args_list.append(args.copy())
    args["subject_1"] = True
    _args_list.append(args.copy())
    args["use_cls"] = False
    _args_list.append(args.copy())
    args["softmax"] = True
    _args_list.append(args.copy())
    return _args_list


if __name__ == '__main__':
    # run in linux cmd
    print("GPU available" if torch.cuda.is_available() else "GPU is not available")
    #"""
    dataset_name = 'semeval' if sys.argv[1] is None else sys.argv[1]
    is_train = True if sys.argv[2] == "t" else False
    if is_train is False:
        sdp_loss_lambda = 0.50
        model_path = sys.argv[3]
    else:
        sdp_loss_lambda = sys.argv[3] if len(sys.argv)<4 else (sys.argv[3],sys.argv[4])
        model_path = None

    # run for testing the lambda
    start_lambda=int(sdp_loss_lambda[0]) if len(sdp_loss_lambda)==2 else sdp_loss_lambda
    end_lambda=int(sdp_loss_lambda[1]) if len(sdp_loss_lambda)==2 else sdp_loss_lambda+1
    #"""
    #model_path = None if sys.argv[3] is None else sys.argv[3]
    #model_path = './ckpt/Newmodel_semeval_sigmoid_withCLS_-2_768_23-06-02_Jun:06:1685681136_bert0.918907.th'

    #model_path = './ckpt/Newmodel_semeval_sigmoid_withCLS_-2_768_lambda0.5_23-06-15_23:45:29_bert.th'
    #model_path = "./ckpt/Newmodel_semeval_sigmoid_withCLS_-2_768_lambda0.5_23-08-02_01:54:02_bert.th"
    #model_path = "./ckpt/Newmodel_semeval_sigmoid_withCLS_-2_768_lambda0.5_23-11-18_08:37:48_bert.th"
    #run in pycharm
    """
    dataset_name='webnlg_star'
    is_train = False
    model_path = "./ckpt/Newmodel_webnlg_star_sigmoid_withCLS_-2_768_lambda0.5_23-12-07_14:16:09_bert.th"
    sdp_loss_lambda = 0.50
    start_lambda,end_lambda = int(sdp_loss_lambda*100), int(sdp_loss_lambda*100+1)
    """

    

    for i in range(start_lambda,end_lambda):
        sdp_loss_lambda=i/100
        task_id = None
        if len(sys.argv) > 3:
            #task_id = int(sys.argv[3])
            task_id = sdp_loss_lambda#/100
        print("==========%s %s============" % (dataset_name, str(task_id)))
        if dataset_name in ["semeval", "semeval_1"]:
            max_epoch = 50
            batch_size = 64
            args_list = get_ablation_args(dataset_name,
                                        max_epoch=max_epoch,
                                        batch_size=batch_size,
                                        cuda_device=2,
                                        # continue_train=True,
                                        # seed=None,
                                        eval=not is_train,
                                        model_path=model_path,
                                        sdp_loss_lambda=sdp_loss_lambda,
                                        #large_bert=True,
                                        )
            train(**args_list[0])

        elif dataset_name in ['kbp37']:
            max_epoch = 50
            batch_size = 20
            args_list = get_ablation_args(dataset_name,
                                        max_epoch=max_epoch,
                                        batch_size=batch_size,
                                        cuda_device=2,
                                        # continue_train=True,
                                        # seed=None,
                                        eval=not is_train,
                                        model_path=model_path,
                                        sdp_loss_lambda=sdp_loss_lambda,
                                        lr=5e-5,
                                        )
            train(**args_list[0])

        elif dataset_name in ['CONLL04']:
            max_epoch = 50
            batch_size = 32
            args_list = get_ablation_args(dataset_name,
                                        max_epoch=max_epoch,
                                        batch_size=batch_size,
                                        cuda_device=2,
                                        # continue_train=True,
                                        # seed=None,
                                        eval=not is_train,
                                        model_path=model_path,
                                        sdp_loss_lambda=sdp_loss_lambda,
                                        lr=5e-5,
                                        )
            train(**args_list[0])

        elif dataset_name in ["nyt10", "nyt10_1", "nyt10_2"]:
            max_epoch = 100
            batch_size = 20
            args_list = get_ablation_args(dataset_name,
                                        max_epoch=max_epoch,
                                        batch_size=batch_size,
                                        cuda_device=3,
                                        lr=5e-5,
                                        sort=False,
                                        eval=not is_train,
                                        model_path=model_path,
                                        sdp_loss_lambda=sdp_loss_lambda,
                                        )
            train(**args_list[0])
        elif dataset_name in ["nyt","nyt24","nyt24_star"]:
            max_epoch = 100
            batch_size = 20
            args_list = get_ablation_args(dataset_name,
                                        max_epoch=max_epoch,
                                        batch_size=batch_size,
                                        cuda_device=3,
                                        lr=5e-5,
                                        sort=False,
                                        eval=not is_train,
                                        model_path=model_path,
                                        sdp_loss_lambda=sdp_loss_lambda,
                                        )
            train(**args_list[0])
        elif dataset_name in ["webnlg_orig", "webnlg", "webnlg_orig_1","webnlg_5019","webnlg_star"]:
            max_epoch = 30
            batch_size = 10
            args_list = get_ablation_args(dataset_name,
                                        max_epoch=max_epoch,
                                        batch_size=batch_size,
                                        sort=False,
                                        cuda_device=3,
                                        continue_train=False,
                                        eval=not is_train,
                                        model_path=model_path,
                                        sdp_loss_lambda=sdp_loss_lambda,
                                        )
            train(**args_list[0])

