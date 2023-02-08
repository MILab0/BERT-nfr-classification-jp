import lib.excel as sheetmaker
import lib.weight_averaging as wa

from tqdm import tqdm
from IPython.display import display
import pandas as pd
import numpy as np
import shutil
import os
import glob
import argparse
import random

import torch
from torch.utils.data import DataLoader
from transformers import CONFIG_NAME, WEIGHTS_NAME, BertForSequenceClassification, BertJapaneseTokenizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import classification_report
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import StochasticWeightAveraging

#MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
MODEL_NAME = 'cl-tohoku/bert-base-japanese-v2'

TRAIN_PATH = 'datasets/random/trainJP_multi.txt'
TEST_PATH = 'datasets/random/testJP_multi.txt'
FOLD = 10
LABEL_LIST = ['A','FT','L','LF','MN','O','PE','PO','SC','SE','US','A_C']
THRESHOLD = -1.25 #default=0

class BertMultiLabelClassifier_pl(pl.LightningModule):

    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_mlc = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            problem_type="multi_label_classification",
            attention_probs_dropout_prob=0.2,
            hidden_dropout_prob=0.2,
        )

    def training_step(self, batch, batch_idx):
        output = self.bert_mlc(
            input_ids=batch['input_ids'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'].to(torch.float)
        )
        loss = output.loss
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self.bert_mlc(
            input_ids=batch['input_ids'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'].to(torch.float)
        )
        val_loss = output.loss
        scores = output.logits
        labels_predicted = (scores > 0).int()
        labels = batch.pop('labels')
        labels.to(torch.float)
        num_correct = ( labels_predicted == labels ).all(-1).sum().item()
        val_acc = num_correct/scores.size(0)
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc)

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        labels.to(torch.float)
        output = self.bert_mlc(**batch)
        scores = output.logits
        labels_predicted = (scores > 0).int()
        num_correct = ( labels_predicted == labels ).all(-1).sum().item()
        accuracy = num_correct/scores.size(0)
        self.log('accuracy', accuracy)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,weight_decay=1e-2)
        return optimizer


class BertModelHandler():
    def __init__(self, data, label_list):
        self.data = data
        self.label_list = label_list
    
    def trainingTaskKFold_ml(self, generalization = False):
        # BERTをファインチューニングして保存する
        if os.path.exists('model_ml/'):
            shutil.rmtree('model_ml/')
    
        if generalization:
            num_of_training = FOLD
        else:
            num_of_training = 1
    
        dataloader_test = DataLoader(self.data['test_data'], batch_size=256)

        #学習データと検証データに分割してファインチューニング
        accuracy = []
        training_data_amounts = []
        val_data_amounts = []
        best_model_paths = []
        #skf = StratifiedKFold(n_splits=FOLD)
        skf = KFold(n_splits=FOLD,shuffle=True)
        for fold, (train_index, val_index) in enumerate(skf.split(X=self.data['dataset'],y=self.data['dataset_idx'])):
        
            train_data = [self.data['dataset'][i] for i in train_index]
            val_data = [self.data['dataset'][i] for i in val_index]
            random.shuffle(train_data)
            training_data_amounts.append(len(train_data))
            val_data_amounts.append(len(val_data))
            dataloader_train = DataLoader(train_data, batch_size=32, shuffle=True)
            dataloader_val = DataLoader(val_data, batch_size=256)
            
            model = BertMultiLabelClassifier_pl(
                MODEL_NAME, num_labels=self.data['num_labels'], lr=8e-6,
            )
            checkpoint = pl.callbacks.ModelCheckpoint(
                filename=f'fold={fold+1}'+'-{epoch}-{step}-{val_loss:.1f}',
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                #save_last=1,
                save_weights_only=True,
                dirpath='model_ml/',
            )
            early_stop = (
                EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    mode='min'
                )
            )
            weight_averaging = (
                StochasticWeightAveraging(swa_lrs=1e-5)
            )
            # 学習方法
            trainer = pl.Trainer(
                gpus=1,
                max_epochs=40,
                log_every_n_steps=10,
                callbacks=[checkpoint, early_stop, weight_averaging]
            )
            # ファインチューニング
            trainer.fit(model,train_dataloaders=dataloader_train,val_dataloaders=dataloader_val)

            print('best model: ', checkpoint.best_model_path)
            print('val_loss: ', checkpoint.best_model_score)
        
            best_model_path = checkpoint.best_model_path

            test = trainer.test(dataloaders=dataloader_test,ckpt_path=best_model_path)
            print(f'Accuracy: {test[0]["accuracy"]:.3f}')
            accuracy.append(test[0]["accuracy"])

            if num_of_training == 1:
                tmp_df = pd.DataFrame({
                    'amount': training_data_amounts,
                    'val_amounts': val_data_amounts
                })
                tmp_df.to_csv('./model_ml/training_data_amounts.csv')
                model = BertMultiLabelClassifier_pl.load_from_checkpoint(best_model_path)
                model.bert_mlc.save_pretrained('./model_transformers_ml/')
                break
            else:
                tmp_df = pd.DataFrame({
                    'amount': training_data_amounts,
                    'val_amounts': val_data_amounts
                })
                tmp_df.to_csv('./model_ml/training_data_amounts.csv')
                best_model_paths.append(best_model_path)
        
        if num_of_training != 1:
            print(f'Average accuracy: {np.mean(accuracy):.3f}')
            print('Starting weight averaging task ...')
            tmp_df = pd.DataFrame({
                'amount': training_data_amounts,
                'val_amounts': val_data_amounts
            })
            tmp_df.to_csv('./model_ml/training_data_amounts.csv')
            #print(tmp_df)
            modelAveraging(best_model_paths,training_data_amounts)
    
    def predictAndEvaluate_ml(self, mode='gpu'):
        #保存済みモデルで予測して評価する
        try:
            # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
            from transformers import logging

            logging.set_verbosity_error()
        except Exception:
            pass

        #保存済みモデルをロード
        bert_mlc = BertForSequenceClassification.from_pretrained(
            './model_transformers_ml/'
        )
        tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
        #符号化
        encoding = tokenizer(
            self.data['test_data_list'],
            padding= 'longest',
            return_tensors='pt'
        )
        if mode == 'gpu':
            #GPUにのせる
            bert_mlc = bert_mlc.cuda()
            encoding = { k: v.cuda() for k, v in encoding.items() }
        else:
            encoding = { k: v for k, v in encoding.items() }
        #予測する
        with torch.no_grad():
            output = bert_mlc(**encoding)
        scores = output.logits #分類スコア
        print(scores)
        labels_predicted = ( scores > THRESHOLD ).int().cpu().numpy().tolist() #予測ラベル(ex.[[1,0],[0,0]])
    
        #CPUに戻す
        #labels_predicted = labels_predicted.cpu()
        scores = scores.cpu()

        #予測Multi-hotラベルを多クラスラベルに変換する
        predicted = []
        for multi_hot in labels_predicted:
            tmp = []
            for idx, true_or_false in enumerate(multi_hot):
                if true_or_false == 1:
                    tmp.append(self.data['index2label'][idx])
            if not tmp:
                predicted.append('OTHER')
            else:
                predicted.append(",".join(tmp))       
    
        #正解Multi-hotラベルを多クラスラベルに変換する
        answer = []
        for multi_hot in self.data['test_data_answer']:
            tmp = []
            for idx, true_or_false in enumerate(multi_hot):
                if true_or_false == 1:
                    tmp.append(self.data['index2label'][idx])
            if not tmp:
                answer.append('OTHER')
            else:
                answer.append(",".join(tmp))

        print('予測ラベル: ',labels_predicted)
        print('正解ラベル: ',self.data['test_data_answer'])

        target_names = self.data['index2label'].values()
        
        report = classification_report(
            self.data['test_data_answer'], labels_predicted, target_names=target_names, output_dict=True, zero_division=0
        )

        report_result = pd.DataFrame(report).T
        display(report_result)

        classification_result = pd.DataFrame({
            'answer_label': answer,
            'predicted_label': predicted,
            'text': self.data['test_data_list'],
        },index=np.arange(1,len(predicted)+1))

        scores_df = pd.DataFrame(scores,columns=self.data['index2label'].values(),index=np.arange(1,len(predicted)+1))

        index2label = self.data['index2label']
        sheetmaker.makeClassificationResultSheet_ml(classification_result,report_result,index2label,scores_df)
        sheetmaker.makeReqAnalysisSheet_ml(classification_result, index2label)


def convert_model(path):
    model = BertMultiLabelClassifier_pl.load_from_checkpoint(path)
    model.bert_mlc.save_pretrained('./model_transformers_ml/')
    print("Model was succesfully saved to model_transformers_ml/")


def modelAveraging(model_paths,amounts):
    #モデルの重みを平均してマージする関数(入力: Pytorch Lightningのチェックポイント 出力: Pytorchモデル)
    try:
        # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
        from transformers import logging

        logging.set_verbosity_error()
    except Exception:
        pass

    models = []
    for path in model_paths:
        model = BertMultiLabelClassifier_pl.load_from_checkpoint(path)
        models.append(model)
    total_amount = sum(amounts)
    training_data_amounts = [c / total_amount for c in amounts]

    base_model = models[0]
    averaged_weights = wa.average_weights(models,training_data_amounts)
    updated_model = wa.update_model(base_model,averaged_weights)
    
    updated_model.bert_mlc.save_pretrained('./model_transformers_ml/')
    print('Model was succesfully saved to model_transformers_ml/')


def loadDatasets_ml(train_path, test_path, label_list):
    train_df = pd.read_csv(f'{train_path}', encoding='cp932')
    test_df = pd.read_csv(f'{test_path}', encoding='cp932')

    index2label = {i: k for i, k in enumerate(label_list)}
    label2index = {k: i for i, k in enumerate(label_list)}
    answers = [test_df[label] for label in label_list]
    label2answers = {label: answers_array for label, answers_array in zip(label_list, answers)}
    
    test_data_list = test_df['feature'].tolist()
    num_labels = len(label_list)

    dataset_for_loader = []
    test_dataset_for_loader = []

    max_length = 128  # トークン数
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

    dataset_idx = []
    for idx, row in tqdm(train_df.iterrows()):
        encoding = tokenizer(
                row['feature'],
                max_length=max_length,
                padding='max_length',
                truncation=True
            )
        encoding['labels'] = [row[label] for label in label_list]  # ラベルを追加
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        dataset_for_loader.append(encoding)
        dataset_idx.append(convertMultihotVector2tmpLabel(encoding['labels'].tolist()))
    
    test_answer = []
    for idx, row in tqdm(test_df.iterrows()):
        encoding = tokenizer(
                row['feature'],
                max_length=max_length,
                padding='max_length',
                truncation=True
            )
        encoding['labels'] = [row[label] for label in label_list]  # ラベルを追加
        test_answer.append([row[label] for label in label_list])
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        test_dataset_for_loader.append(encoding)

    return {
        'num_labels': num_labels,
        'index2label': index2label,
        'label2index': label2index,
        'dataset': dataset_for_loader,
        'dataset_idx': dataset_idx,
        'test_data': test_dataset_for_loader,
        'test_data_answer': test_answer,
        'label2answers': label2answers,
        'test_data_list': test_data_list,
    }


def convertMultihotVector2tmpLabel(mlt_hot):
    #one-hot形式のラベルを一時的なマルチクラスのラベルに変換する
    length = len(mlt_hot)
    label = None
    for idx, single_one_hot_vector in enumerate(np.identity(length,dtype=int).tolist()):
        if single_one_hot_vector == mlt_hot:
            label = idx
    if label == None:
        label = length + 1 #Other
    return label

#--skip_training で学習ステップをスキップする
#--gen で学習をfold回行う
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--skip_training",help='optional',action='store_true')
    argparser.add_argument("--gen",help='optional',action='store_true')
    argparser.add_argument("--merge_checkpoint",help='optional',action='store_true')
    argparser.add_argument("--convert_model",type=str,help='Path to directory convert .ckpt to .bin',required=False)
    args = argparser.parse_args()

    print('Is CUDA available?: ',torch.cuda.is_available())
    mode = 'gpu'
    if torch.cuda.is_available() is False:
        if not args.skip_training:
            print("Can't TRAIN without GPU. Enable GPU acceleration and try again.")
            exit()
        print("Start prediction task without GPU ...")
        mode = 'cpu'

    data = loadDatasets_ml(TRAIN_PATH, TEST_PATH, LABEL_LIST)
    bmh = BertModelHandler(data, LABEL_LIST)

    if args.skip_training:
        print('Training task skipped.')
        bmh.predictAndEvaluate_ml(mode=mode)
    elif args.gen:
        print(f'Starting training task with {FOLD} fold ...')
        bmh.trainingTaskKFold_ml(generalization=True)
        bmh.predictAndEvaluate_ml()
    elif args.merge_checkpoint:
        print('Starting model merging task ...')
        models_path = glob.glob("./model_ml/*.ckpt")
        amount_df = pd.read_csv("./model_ml/training_data_amounts.csv")
        amounts = amount_df['amount'].tolist()
        total = sum(amounts)
        training_data_amounts = [c / total for c in amounts]
        modelAveraging(models_path,training_data_amounts)
    elif args.convert_model:
        print("Convert .ckpt model to pytorch model ...")
        convert_model(f'./model/{args.convert_model}')
    else:
        print('Starting training task...')
        bmh.trainingTaskKFold_ml()
        bmh.predictAndEvaluate_ml()