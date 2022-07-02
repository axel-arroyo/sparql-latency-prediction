from sklearn.preprocessing import StandardScaler
import pandas as pd
URL = "~/Desktop/"
import torch.nn as nn
from sklearn.model_selection import train_test_split
from Models.model_trees_algebra import NeoRegression


class BaoTrainingException(Exception):
    pass


def split_ds(all_data, val_rate, seed):
    """test_rate is a rate of the total, val_rate is a rate of the (total - test_rate)"""
    ranges = {}
    ranges['1_2'] = all_data[(all_data["time"] > 0)    & (all_data["time"] <= 2)]
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


ds_test = pd.read_csv(URL + "ds_test.csv", delimiter="ᶶ", engine='python').sample(1000)
# ds_train = pd.read_csv(URL + "ds_train.csv", delimiter="ᶶ", engine='python')
data_train_val = pd.read_csv(URL + "ds_trainval.csv", delimiter="ᶶ", engine='python')

ds_trainval = data_train_val[data_train_val['time'] <=65].sample(3000)
# ds_val = ds_val[ds_val['time'] <=65]
ds_test = ds_test[ds_test['time'] <=65]
list_columns = ['filter_bound', 'filter_contains', 'filter_eq', 'filter_exists',
       'filter_ge', 'filter_gt', 'filter_isBlank', 'filter_isIRI',
       'filter_isLiteral', 'filter_lang', 'filter_langMatches', 'filter_le',
       'filter_lt', 'filter_ne', 'filter_not', 'filter_notexists', 'filter_or',
       'filter_regex', 'filter_sameTerm', 'filter_str', 'filter_strends',
       'filter_strstarts', 'filter_subtract', 'has_slice', 'max_slice_limit', 'max_slice_start','json_cardinality']
x_test_query  = ds_test[list_columns]
x_test_tree = ds_test['trees'].values
y_test = ds_test['time'].values
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

x_train_tree = ds_train['trees'].values
x_val_tree = ds_val['trees'].values

y_train = ds_train['time'].values
y_val = ds_val['time'].values

import json
def getmax(x):
    lista=  list(x.values())
    maximo = 0
    for el in lista:
        if (maximo < float(el)):
            maximo = float(el)
    return maximo
# x_train_query['json_cardinality'] = x_train_query['json_cardinality'].apply(lambda x: json.loads(x))
maximotrain =  x_train_query['json_cardinality'].apply(lambda x: json.loads(x)).apply(lambda x: getmax(x)).max()





aec_dir = '/workspace/bao_server/aec_wikidata_neo.pth'
verbose=True
reg = NeoRegression(
     aec={'train_aec': False, 'use_aec': True,'aec_file': aec_dir+str(fold)+'.pth', 'aec_epochs': 200},
     epochs=400,
     maxcardinality=maximotrain,
     in_channels_neo_net=512,
     tree_units=[1024, 512, 256, 128],
     tree_units_dense=[64, 32],
     early_stop_patience=10,
     early_stop_initial_patience=180,
     tree_activation_tree=nn.ReLU,
     tree_activation_dense=nn.ReLU,
    optimizer={'optimizer': "Adam",'args':{"lr":0.00015}},
    figimage_size=(18,18)
)


#Fit the transformer tree data
reg.fit_transform_tree_data(ds_train, ds_val, ds_test)
print("Trees tranformed!!!")

def getmax(x):
    lista=  list(x.values())
    maximo = 0
    for el in lista:
        if (maximo < float(el)):
            maximo = float(el)
    return maximo
# x_train_query['json_cardinality'] = x_train_query['json_cardinality'].apply(lambda x: json.loads(x))
maximotrain =  x_train_query['json_cardinality'].apply(lambda x: json.loads(x)).apply(lambda x: getmax(x)).max()
#Scale x_query data.

xqtrain = x_train_query.drop(columns=['json_cardinality'])
xqval   = x_val_query.drop(columns=['json_cardinality'])
xqtest  = x_test_query.drop(columns=['json_cardinality'])

scalerx = StandardScaler()
x_train_scaled = scalerx.fit_transform(xqtrain)
x_val_scaled = scalerx.transform(xqval)
x_test_scaled = scalerx.transform(xqtest)

x_train_query =pd.concat([ pd.DataFrame(x_train_scaled, index=xqtrain.index, columns=xqtrain.columns),x_train_query[['json_cardinality']]], axis=1)
x_val_query = pd.concat([pd.DataFrame(x_val_scaled, index=xqval.index, columns=xqval.columns),x_val_query[['json_cardinality']]], axis=1)
x_test_query = pd.concat([pd.DataFrame(x_test_scaled, index=xqtest.index, columns=xqtest.columns),x_test_query[['json_cardinality']]], axis=1)

def pred2index_dict(x, pred_to_index):
    resp = {}
    x = json.loads(x)
    for el in x.keys():
        if el in pred_to_index:
            resp[pred_to_index[el]] = float(x[el])/maximotrain
    return resp
x_train_query['json_cardinality'] = x_train_query['json_cardinality'].apply(lambda x: pred2index_dict(x, reg.get_pred()))

del xqtrain
del xqval
del xqtest
del x_train_scaled
del x_val_scaled
del x_test_scaled
#Fit model
reg.fit(x_train_tree, x_train_query.values, y_train, x_val_tree, x_val_query.values, y_val)
