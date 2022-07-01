from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from model_trees import TreeRegression

URL = "/workspace/TreeConvolution/DS_WIKIDATA_TESIS/"

import torch.nn as nn

class BaoTrainingException(Exception):
    pass
from sklearn.model_selection import train_test_split
def split_ds(all_data, val_rate, seed):
    """test_rate is a rate of the total, val_rate is a rate of the (total - test_rate)"""
    ranges = {}
    ranges['1_2'] = all_data[(all_data["time"] > 0)    & (all_data["time"] <= 2)]
    print(ranges['1_2'].shape)
    ranges['2_3'] = all_data[(all_data["time"] > 2)    & (all_data["time"] <= 3)]
    ranges['3_4'] = all_data[(all_data["time"] > 3)    & (all_data["time"] <= 4)]
    ranges['4_5'] = all_data[(all_data["time"] > 4)    & (all_data["time"] <= 5)]
    ranges['5_8'] = all_data[(all_data["time"] > 5)    & (all_data["time"] <= 8)]
    ranges['8_10'] = all_data[(all_data["time"] > 8)   & (all_data["time"] <= 10)]
    ranges['10_20'] =   all_data[(all_data["time"] > 10) & (all_data["time"] <= 20)]
    ranges['20_30'] =   all_data[(all_data["time"] > 20) & (all_data["time"] <= 30)]
    ranges['30_40'] =   all_data[(all_data["time"] > 30) & (all_data["time"] <= 40)]
    ranges['40_50'] =   all_data[(all_data["time"] > 40) & (all_data["time"] <= 50)]
    ranges['50_60'] =   all_data[(all_data["time"] > 50) & (all_data["time"] <= 60)]
    ranges['60_80'] =   all_data[(all_data["time"] > 60) & (all_data["time"] <= 80)]
    ranges['80_100'] =  all_data[(all_data["time"] > 80) & (all_data["time"] <= 100)]
    ranges['100_150'] = all_data[(all_data["time"] > 100) & (all_data["time"] <= 150)]
    ranges['150_last'] = all_data[(all_data["time"] > 150)]
    train_data = []
    val_data = []
    test_data = []
    for rang in ranges.values():
        if rang.shape[0] >= 3:
            X_train, X_val = train_test_split(
                rang, test_size=val_rate, shuffle=True,random_state=seed)

            train_data.append(X_train)
            val_data.append(X_val)
    train_data_list = pd.concat(train_data)
    val_data_list = pd.concat(val_data)
    print("Shapes : Train: {} Val: {}".format(train_data_list.shape, val_data_list.shape))
    return train_data_list, val_data_list


ds_test = pd.read_csv(URL + "ds_ready_test_4239.csv", delimiter="ᶶ", engine='python')
# ds_train = pd.read_csv(URL + "ds_train.csv", delimiter="ᶶ", engine='python')
data_train_val = pd.read_csv(URL + "ds_ready_train_31042.csv", delimiter="ᶶ", engine='python')

ds_trainval = data_train_val[data_train_val['time'] <=65]
# ds_val = ds_val[ds_val['time'] <=65]
ds_test = ds_test[ds_test['time'] <=65]

x_test_tree = ds_test['trees'].values
y_test = ds_test['time'].values
print(np.max(y_test))

print("Shape: train_data" , ds_trainval.shape)
rmsesss = []
predictions = []
fold = 0
#Datasets
ds_train, ds_val = split_ds(ds_trainval, 0.2, seed=fold)

x_train_tree = ds_train['trees'].values
x_val_tree = ds_val['trees'].values

y_train = ds_train['time'].values
y_val = ds_val['time'].values

aec_dir = '/workspace/bao_server/aec_wikidata.pth'
verbose=True

reg = TreeRegression(
     aec={'train_aec': True, 'use_aec': True,'aec_file': aec_dir+str(fold)+'.pth', 'aec_epochs': 200},
     epochs=200,
     in_channels_neo_net=512,
     tree_units=[512, 256, 128],
     tree_units_dense=[64, 32],
     early_stop_patience=10,
     early_stop_initial_patience=80,
     tree_activation_tree=nn.Tanh,
     tree_activation_dense=nn.ReLU,
    optimizer={'optimizer': "Adam",'args':{"lr":0.00015}},
    figimage_size=(18,18)
)

#Fit the transformer tree data
reg.fit_transform_tree_data(x_train_tree, x_val_tree, x_test_tree)
print("Trees tranformed!!!")
#Fit model
reg.fit(x_train_tree, y_train, x_val_tree, y_val)

x_test_tree = reg.json_loads(x_test_tree)
x_test_tree = [reg.fix_tree(x)  for x in x_test_tree]
x_test_tree = reg.transform_trees(x_test_tree)

preds_test2 = reg.predict_raw_data(x_test_tree)

rmsetest2 = np.sqrt(mean_squared_error(y_test, preds_test2))
print("RMSE in fold:{}, set TEST: {}".format(fold, rmsetest2))
reg.scatter_image(preds_test2, y_test, "Scatter real latency vs prediction on Test dataset.", "neo_with_aec_scatter_test",max_refference=70,figsize =(15,15))
rmsesss.append(rmsetest2)
predictions.append(preds_test2)
