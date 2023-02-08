import argparse
import glob
import os
import sys
import pandas as pd
from transformers import CONFIG_NAME, WEIGHTS_NAME, AutoModelForSequenceClassification
import torch
from collections import OrderedDict
from tqdm import tqdm

def average_weights(input_models, coefficients):
    weights_averaged = OrderedDict()
    for i, current_model in tqdm(enumerate(input_models), leave=False):
        current_weights = current_model.state_dict()
        for key in current_weights.keys():
            if i == 0:
                weights_averaged[key] = coefficients[i] * current_weights[key]
            else:
                weights_averaged[key] += coefficients[i] * current_weights[key]

    return weights_averaged


def update_model(base_model, weights):

    base_model.load_state_dict(weights)

    return base_model


class ModelAvaraging():

    def __init__(self,model_paths,amounts):
        self.model_paths = model_paths
        self.amounts = amounts

    #モデルの重みを平均化するメソッド(Pytorchのモデルを使う)
    def sequenceClassifierAveraging_pytorch(self,save_dir):
        models = []
        for path in self.model_paths:
            model = AutoModelForSequenceClassification.from_pretrained(path)
            models.append(model)
        total_amount = sum(self.amounts)
        training_data_amounts = [c / total_amount for c in self.amounts]

        base_model = models[0]
        averaged_weights = average_weights(models,training_data_amounts)
        updated_model = update_model(base_model,averaged_weights)
    
        final_model_file = os.path.join(f'./{save_dir}/',WEIGHTS_NAME)
        torch.save(updated_model.state_dict(),final_model_file)
        output_config_file = os.path.join(f'./{save_dir}/',CONFIG_NAME)
        with open(output_config_file,'w') as f:
            f.write(updated_model.config.to_json_string())
        print(f'Model was succesfully saved to {save_dir}/ .')

#PytorchのTransformerモデルの重みを平均化したモデルを生成します。
#使い方: python lib/weight_averaging.py --target_dir 複数のPytorchモデルが格納されたフォルダ名 --save_dir 出力するフォルダ名
#※各モデル学習時の学習データの数を記載したファイル'training_data_amounts.csv'が必要です。
#training_data_amounts.csvにカラム名'amount'、行に学習データの数を記載してtarget_dirの中に入れて下さい。
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--target_dir','--save_dir',type=str)
    args = argparser.parse_args()

    model_paths = glob.glob(args.target_dir)
    amounts_file = os.path.join(f'./{args.target_dir}/','training_data_amounts.csv')
    if os.path.exists(amounts_file) == False:
        print("Error: 各モデル学習時の学習データの数を記載したファイル'training_data_amounts.csv'が必要です。\ntraining_data_amounts.csvにカラム名'amount'、行に学習データの数を記載してtarget_dirの中に入れて下さい。",file=sys.stderr)
        sys.exit(1)
    amount_df = pd.read_csv(amounts_file)
    amounts = amount_df['amount'].tolist()
    ma = ModelAvaraging(model_paths,amounts)

    ma.sequenceClassifierAveraging_pytorch(args.save_dir)