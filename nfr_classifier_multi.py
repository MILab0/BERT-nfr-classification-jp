import lib.excel as sheetmaker

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
from transformers import CONFIG_NAME, WEIGHTS_NAME, BertModel, BertJapaneseTokenizer
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
THRESHOLD = 0 #default=0

class BertMultiLablelClassifier(torch.nn.Module):

    def __init__(self, model_name, num_labels):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)

        self.linear = torch.nn.Linear(
            self.bert.config.hidden_size, num_labels
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        #BERTの最終層の出力
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        last_hidden_state = bert_output.last_hidden_state
        #[PAD]トークン以外で隠れ状態の平均をとる
        averaged_hidden_state = \
            (last_hidden_state*attention_mask.unsqueeze(-1)).sum(1) \
            / attention_mask.sum(1, keepdim=True)
        
        #線形変換
        scores = self.linear(averaged_hidden_state)
        output = {'logits': scores}

        if labels is not None:
            loss = torch.nn.BCEWithLogitsLoss()(scores, labels.float())
            output['loss'] = loss

        output = type('bert_output', (object,), output)
        return output


class BertMultiLabelClassifier_pl(pl.LightningModule):

    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_mlc = BertMultiLablelClassifier(
            model_name, 
            num_labels=num_labels,
        )

    def training_step(self, batch, batch_idx):
        output = self.bert_mlc(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self.bert_mlc(**batch)
        val_loss = output.loss
        scores = output.logits
        labels_predicted = (scores > 0).int()
        labels = batch.pop('labels')
        num_correct = ( labels_predicted == labels ).all(-1).sum().item()
        val_acc = num_correct/scores.size(0)
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc)

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
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
    
    # BERTをファインチューニングして保存する
    def trainingTaskKFold_ml(self, generalization = False):
        if os.path.exists('model_ml/'):
            shutil.rmtree('model_ml/')
    
        if generalization:
            num_of_training = FOLD
        else:
            num_of_training = 1
    
        dataloader_test = DataLoader(self.data['test_data'], batch_size=256)

        #層化K分割で学習データと検証データに分割してファインチューニング
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
                MODEL_NAME, num_labels=self.data['num_labels'], lr=1e-5,
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
    
    #保存済みモデルで予測して評価する
    def predictAndEvaluate_ml(self, mode='gpu'):
        try:
            # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.
            from transformers import logging

            logging.set_verbosity_error()
        except Exception:
            pass

        #保存済みモデルをロード
        dir_path = './model_ml/*ckpt'
        model_path = glob.glob(dir_path)
        model = BertMultiLabelClassifier_pl.load_from_checkpoint(
            model_path[0]
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
            if type(model) == BertMultiLabelClassifier_pl:
                bert_mlc = model.bert_mlc.cuda()
            else:
                bert_mlc = model.cuda()
            encoding = { k: v.cuda() for k, v in encoding.items() }
        else:
            if type(model) == BertMultiLabelClassifier_pl:
                bert_mlc = model.bert_mlc
            else:
                bert_mlc = model
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

#one-hot形式のラベルを一時的なマルチクラスのラベルに変換する
def convertMultihotVector2tmpLabel(mlt_hot):
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
    args = argparser.parse_args()

    print('Is CUDA available?: ',torch.cuda.is_available())
    mode = 'gpu'
    if torch.cuda.is_available() == False:
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
    else:
        print('Starting training task...')
        bmh.trainingTaskKFold_ml()
        bmh.predictAndEvaluate_ml()