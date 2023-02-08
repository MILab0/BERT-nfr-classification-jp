import lib.excel as sheetmaker
import lib.weight_averaging as wa
import lib.model_explainer as me

from tqdm import tqdm
from IPython.display import display
import pandas as pd
import numpy as np
import shutil
import glob
import os
import argparse
import random

import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import StochasticWeightAveraging

#MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
MODEL_NAME = 'cl-tohoku/bert-base-japanese-v2'

TRAIN_PATH = 'datasets/random/trainJP_nfr.txt'
TEST_PATH = 'datasets/random/testJP_nfr.txt'
FOLD = 10
# ---------------------------------------------------------------------------

class BertClassifier_pl(pl.LightningModule):

    def __init__(self, model_name, num_labels, lr, train_batch_size = 32):
        #model_name: Transformersのモデル名
        #num_labels: ラベルの数
        #lr: 学習率
        #train_batch_size: 学習データのバッチサイズ
        super().__init__()

        self.save_hyperparameters()

        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            #attention_probs_dropout_prob=0.2,
            #hidden_dropout_prob=0.2,
        )
        
    # 学習データのミニバッチの損失を出力
    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss

    # 検証データのミニバッチの損失,精度を出力
    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        labels = batch.pop('labels')
        labels_predicted = output.logits.argmax(-1)
        num_correct = (labels_predicted == labels).sum().item()
        val_acc = num_correct/labels.size(0)
        self.log('val_loss', val_loss)
        self.log('val_acc',val_acc)
        return {"val_loss": val_loss,"val_acc": val_acc}

    # テストデータのミニバッチの精度を出力
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        num_correct = (labels_predicted == labels).sum().item()
        accuracy = num_correct/labels.size(0)
        self.log('accuracy', accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,weight_decay=1e-2)
        #optimizer = AdamW(self.parameters(),lr=self.hparams.lr,weight_decay=1e-2)
        return [optimizer]

class BertModelHandler():
    def __init__(self, data):
        self.data = data
    
    # BERTをファインチューニングして保存する
    def trainingTaskKFold(self, generalization = False):
        if os.path.exists('model/'):
            shutil.rmtree('model/')
    
        if generalization is True:
            num_of_training = FOLD
        else:
            num_of_training = 1
    
        dataloader_test = DataLoader(self.data['test_data'], batch_size=256)

        #層化K分割で学習データと検証データに分割してファインチューニング
        accuracy = []
        best_model_paths = []
        training_data_amounts = []
        val_data_amounts = []
        skf = StratifiedKFold(n_splits=FOLD)
        for fold, (train_index, val_index) in enumerate(skf.split(X=self.data['dataset'],y=self.data['dataset_idx'])):
        
            train_data = [self.data['dataset'][i] for i in train_index]
            val_data = [self.data['dataset'][i] for i in val_index]
            #print(f'TRAIN:{train_index} VAL:{val_index}')
            random.shuffle(train_data)
            training_data_amounts.append(len(train_data))
            val_data_amounts.append(len(val_data))
            dataloader_train = DataLoader(train_data, batch_size=32, shuffle=True)
            dataloader_val = DataLoader(val_data, batch_size=256)

            model = BertClassifier_pl(
                MODEL_NAME, num_labels=self.data['num_labels'], lr=1e-5,
            )
            checkpoint = pl.callbacks.ModelCheckpoint(
                filename=f'fold={fold+1}'+'-{epoch}-{step}-{val_loss:.1f}',
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                #save_last=1,
                save_weights_only=True,
                dirpath='model/',
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
                model = BertClassifier_pl.load_from_checkpoint(best_model_path)
                model.bert_sc.save_pretrained('./model_transformers/')
                tmp_df = pd.DataFrame({
                'amount': training_data_amounts,
                'val_amounts': val_data_amounts
                })
                tmp_df.to_csv('./model/training_data_amounts.csv')
                break
            else:
                best_model_paths.append(best_model_path)
        
        if num_of_training != 1:
            print(f'Average accuracy: {np.mean(accuracy):.3f}')
            print('Starting weight averaging task ...')
            tmp_df = pd.DataFrame({
                'amount': training_data_amounts,
                'val_amount': val_data_amounts
            })
            tmp_df.to_csv('./model/training_data_amounts.csv')
            modelAveraging(best_model_paths,training_data_amounts)
        
    #保存済みモデルで予測して評価する
    def predictAndEvaluate(self, mode='gpu'):
        #保存済みモデルをロード
        bert_sc = BertForSequenceClassification.from_pretrained(
            './model_transformers/'
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
            bert_sc = bert_sc.cuda()
            encoding = { k: v.cuda() for k, v in encoding.items() }
        else:
            encoding = { k: v for k, v in encoding.items() }

        #予測する
        with torch.no_grad():
            output = bert_sc.forward(**encoding)
        scores = output.logits #分類スコア
        print(scores)
        labels_predicted = scores.argmax(-1) #スコアが最も高いラベルインデックス
        labels_predicted_2nd = scores.argsort()[:,-2] #スコアが2番目に高いラベルインデックス
        labels_predicted_3rd = scores.argsort()[:,-3] #スコアが3番目に高いラベルインデックス
        #CPUに戻す
        if mode == 'gpu':
            labels_predicted = labels_predicted.cpu()
            labels_predicted_2nd = labels_predicted_2nd.cpu()
            labels_predicted_3rd = labels_predicted_3rd.cpu()
            scores = scores.cpu()

        score_max = []
        score_min = []
        for score_array in scores:
            score_min.append(score_array.min())
            score_max.append(score_array.max())
        score_diff = []
        for min,max in zip(score_min,score_max):
            score_diff.append(abs(max-min))

        #予測インデックスをラベルに変換する
        predicted = ['UNKNOWN' if diff < 0.1 else self.data['index2label'][index] for index, diff in zip(labels_predicted.tolist(),score_diff)]
        predicted_2nd = ['UNKNOWN' if diff < 0.1 else self.data['index2label'][index] for index, diff in zip(labels_predicted_2nd.tolist(),score_diff)]
        predicted_3rd = ['UNKNOWN' if diff < 0.1 else self.data['index2label'][index] for index, diff in zip(labels_predicted_3rd.tolist(),score_diff)]

        print('予測ラベル: ',predicted)
        print('正解ラベル: ',self.data['test_data_answer'])

        target_names = list(self.data['index2label'].values())
        label_ids = list(self.data['index2label'].keys())

        #正解ラベルの確認
        for idx, label in enumerate(self.data['test_data_answer']):
            if label not in self.data['label2index'].keys():
                self.data['test_data_answer'][idx] = label+' (UNKNOWN)'
            
        #正解ラベルをインデックスに変換する
        ans_labels = []
        for label in self.data['test_data_answer']:
            if label not in self.data['label2index'].keys():
                ans_labels.append(99) #unknown
            else:
                ans_labels.append(self.data['label2index'][label])

        report = classification_report(
            ans_labels, labels_predicted, labels=label_ids, target_names=target_names, output_dict=True, zero_division=0
        )

        report_result = pd.DataFrame(report).T
        display(report_result)

        classification_result = pd.DataFrame({
            'answer_label': self.data['test_data_answer'], 
            'predicted_label': predicted,
            '2nd_predicted': predicted_2nd,
            '3rd_predicted': predicted_3rd,
            'text': self.data['test_data_list'],
        },index=np.arange(1,len(predicted)+1))

        scores_df = pd.DataFrame(scores,columns=self.data['index2label'].values(),index=np.arange(1,len(predicted)+1))

        index2label = self.data['index2label']
        sheetmaker.makeClassificationResultSheet(classification_result,report_result,index2label,scores_df)

        #modelExplainer = me.ModelExplainer(model=predict,tokenizer=tokenizer,labels=target_names)
        #modelExplainer.shap_explainer(self.data['test_data_list'],self.data['test_data_answer'],predicted)

# データセットを読み込んでBERTに入力可能な形式に変換する
# dataset: 分割前データセット, dataset_idx: 分割前データセットのラベル
def loadDatasets(train_path, test_path):
    train_df = pd.read_csv(f'{train_path}', encoding='cp932')
    test_df = pd.read_csv(f'{test_path}', encoding='cp932')

    labels = {i: k for i, k in enumerate(train_df['label'].unique()) }
    index2label = {i: k for i, k in enumerate(labels.values())}
    label2index = {k: i for i, k in enumerate(labels.values())}

    test_label_answer = test_df['label'].tolist()
    test_data_list = test_df['feature'].tolist()
    category_list = index2label.values()
    num_labels = len(category_list)

    dataset_for_loader = []
    test_dataset_for_loader = []

    max_length = 128  # トークン数
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

    dataset_idx = [] #ラベル
    for idx, row in tqdm(train_df.iterrows()):
        encoding = tokenizer(
                row['feature'],
                max_length=max_length,
                padding='max_length',
                truncation=True
            )
        encoding['labels'] = label2index[row['label']]  # ラベルを追加
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        dataset_for_loader.append(encoding)
        dataset_idx.append(label2index[row['label']])
    
    for idx, row in tqdm(test_df.iterrows()):
        encoding = tokenizer(
                row['feature'],
                max_length=max_length,
                padding='max_length',
                truncation=True
            )
        encoding['labels'] = label2index[row['label']]  # ラベルを追加
        encoding = {k: torch.tensor(v) for k, v in encoding.items()}
        test_dataset_for_loader.append(encoding)

    return {
        'num_labels': num_labels,
        'index2label': index2label,
        'label2index': label2index,
        'dataset': dataset_for_loader,
        'dataset_idx': dataset_idx,
        'test_data': test_dataset_for_loader,
        'test_data_answer': test_label_answer,
        'test_data_list': test_data_list,
    }


def modelAveraging(model_paths,amounts):
    #モデルの重みを平均してマージする関数(入力: Pytorch Lightningのチェックポイント 出力: Pytorchモデル)
    models = []
    for path in model_paths:
        model = BertClassifier_pl.load_from_checkpoint(path)
        models.append(model)
    total_amount = sum(amounts)
    training_data_amounts = [c / total_amount for c in amounts]

    base_model = models[0]
    averaged_weights = wa.average_weights(models,training_data_amounts)
    updated_model = wa.update_model(base_model,averaged_weights)
    
    updated_model.bert_sc.save_pretrained('./model_transformers/')
    print('Model was succesfully saved to model_transformers/')

def convert_model(path):
    model = BertClassifier_pl.load_from_checkpoint(path)
    model.bert_sc.save_pretrained('./model_transformers/')
    print("Model was succesfully saved to model_transformers/")

def predict(features):
    #保存済みモデルをロード
    bert_sc = BertForSequenceClassification.from_pretrained(
        './model_transformers/'
    )
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
    #符号化
    encoding = tokenizer(
        list(features),
        padding= 'longest',
        return_tensors='pt'
    )
    bert_sc.cuda()
    encoding = { k: v.cuda() for k, v in encoding.items() }
    with torch.no_grad():
        output = bert_sc.forward(**encoding)
    return output.logits.cpu()

# -----------------------------------------------------------------------------

#--skip_training で学習ステップをスキップする
#--gen で学習をfold回行う
if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--skip_training",help='optional',action='store_true')
    argparser.add_argument("--gen",help='optional',action='store_true')
    argparser.add_argument("--weight_averaging_only",help='optional',action='store_true')
    argparser.add_argument("--convert_model",help='optional',type=str)
    args = argparser.parse_args()

    print('Is CUDA available?: ',torch.cuda.is_available())
    mode = 'gpu'
    if torch.cuda.is_available() is False:
        if not args.skip:
            print("Can't TRAIN without GPU. Enable GPU acceleration and try again.")
            exit()
        print("Start prediction task without GPU ...")
        mode = 'cpu'

    data = loadDatasets(TRAIN_PATH, TEST_PATH)
    bmh = BertModelHandler(data)

    if args.skip_training:
        print('Training task skipped.')
        bmh.predictAndEvaluate(mode)
    elif args.gen:
        print(f'Starting training task with {FOLD} fold ...')
        bmh.trainingTaskKFold(generalization=True)
        bmh.predictAndEvaluate()
    elif args.weight_averaging_only:
        print('Starting weight averaging task ...')
        models_path = glob.glob("./model/*.ckpt")
        amount_df = pd.read_csv("./model/training_data_amounts.csv")
        amounts = amount_df['amount'].tolist()
        total = sum(amounts)
        training_data_amounts = [c / total for c in amounts]
        modelAveraging(models_path,training_data_amounts)
    elif args.convert_model:
        print("Convert .ckpt model to pytorch model ...")
        convert_model(f'./model/{args.convert_model}')
    else:
        print('Starting training task...')
        bmh.trainingTaskKFold()
        bmh.predictAndEvaluate()