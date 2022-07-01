from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from model_dense import DenseRegression

URL = "~/Desktop/"
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
list_columns = ['triple', 'bgp', 'join', 'leftjoin', 'union', 'filter', 'graph',
                'extend', 'minus', 'path*', 'pathN*', 'path+', 'pathN+', 'path?',
                'notoneof', 'tolist', 'order', 'project', 'distinct', 'reduced',
                'multi', 'top', 'group', 'assign', 'sequence', 'slice', 'treesize',
                'pcs0', 'pcs1', 'pcs2', 'pcs3', 'pcs4', 'pcs5', 'pcs6', 'pcs7',
                'pcs8', 'pcs9', 'pcs10', 'pcs11', 'pcs12', 'pcs13', 'pcs14',
                'pcs15', 'pcs16', 'pcs17', 'pcs18', 'pcs19', 'pcs20', 'pcs21',
                'pcs22', 'pcs23', 'pcs24']
x_test_query  = ds_test[list_columns]
y_test = ds_test['time'].values
print(np.max(y_test))

print("Shape: train_data" , ds_trainval.shape)
rmsesss = []
predictions = []
fold = 0
#Datasets
ds_train, ds_val = split_ds(ds_trainval, 0.2, seed=fold)
#     ds_train = resampling(ds_train)
#     X_train, X_test, y_train, y_test = train_test_split(data_train_val[list_columns], data_train_val['trees'], test_size=0.33, random_state=42)
x_train_query = ds_train[list_columns]
x_val_query   = ds_val[list_columns]

y_train = ds_train['time'].values
y_val = ds_val['time'].values


scalerx = StandardScaler()
x_train_scaled = scalerx.fit_transform(x_train_query)
x_val_scaled = scalerx.transform(x_val_query)
x_test_scaled = scalerx.transform(x_test_query)

#Scale x_query data.
scaled_df_train = pd.DataFrame(x_train_scaled, index=x_train_query.index, columns=x_train_query.columns)
scaled_df_val = pd.DataFrame(x_val_scaled, index=x_val_query.index, columns=x_val_query.columns)
scaled_df_test = pd.DataFrame(x_test_scaled, index=x_test_query.index, columns=x_test_query.columns)

aec_dir = '/workspace/bao_server/aec_wikidata.pth'
verbose=True
reg = DenseRegression(
    verbose=True,
     epochs=250,
     query_input_size= scaled_df_train.shape[0],
     query_hidden_inputs=[260,380,240],
     query_output=1,
     # loss={'func':'CustomMSELoss', 'threshold': 50, 'weight_factor': 1},
     early_stop_patience=10,
     early_stop_initial_patience=80,
     activation_dense=nn.ReLU,
    optimizer={'optimizer': "Adam",'args':{"lr":0.00015}},
    figimage_size=(18,18)
)

reg.fit(scaled_df_train.values, y_train, scaled_df_val.values, y_val)


preds_test2 = reg.predict_raw_data(scaled_df_test.values)
rmsetest2 = np.sqrt(mean_squared_error(y_test, preds_test2))
print("RMSE in fold:{}, set TEST: {}".format(fold, rmsetest2))
reg.scatter_image(preds_test2, y_test, "Scatter real latency vs prediction on Test dataset.", "neo_with_aec_scatter_test",max_refference=70,figsize =(15,15))
rmsesss.append(rmsetest2)
predictions.append(preds_test2)
