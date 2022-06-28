from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import model
import numpy as np
import pandas as pd
URL = "/workspace/TreeConvolution/DS_WIKIDATA_TESIS/"
import matplotlib.pyplot as plt

class BaoTrainingException(Exception):
    pass


def train_and_save_model(fn, verbose=True, emphasize_experiments=0):

    ds_train = pd.read_csv(URL + "ds_train_resampled.csv", delimiter="ᶶ",engine='python')
    ds_val = pd.read_csv(URL + "ds_val.csv", delimiter="ᶶ",engine='python')
    ds_test = pd.read_csv(URL + "ds_test.csv", delimiter="ᶶ",engine='python')
    x_train = ds_train['trees'].values
    y_train = ds_train['time'].values
    x_val = ds_val['trees'].values
    y_val = ds_val['time'].values
    x_test = ds_test['trees'].values
    y_test = ds_test['time'].values
    
    reg = model.BaoRegression(have_cache_data=False, aec_file=None, epochs=200, verbose=verbose)
    #Fit the transformer tree data
    reg.fit_transform_tree_data(x_train, x_val, x_test)
    
    losses = reg.fit(x_train, y_train, x_val, y_val)
    plt.plot(losses['train_loss'],label="train")
    plt.plot(losses['val_loss'],label="val")
    plt.savefig('loss_training_bao.png')
    plt.legend()
    plt.show()
    reg.save(fn)
    preds_val = reg.predict(ds_val['trees'].values)
    y_val = ds_val['time'].values
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    print("RMSE in VAL: {}".format(rmse))
    reg.scatter_image(preds_val, y_val, "Scatter real latency vs prediction on Val dataset.", "bao_with_aec_scatter_val")
    y_test = ds_test['time'].values
    preds_test = reg.predict(ds_test['trees'].values)
    rmsetest = np.sqrt(mean_squared_error(y_test, preds_test))
    print("RMSE in TEST: {}".format(rmsetest))
    reg.scatter_image( preds_test, y_test, "Scatter real latency vs prediction on Test dataset.", "bao_with_aec_scatter_test")

    return reg


def train_and_save_neo_model(fn, aec_dir=None, verbose=True, emphasize_experiments=0):
    # all_experience = experience()
    #
    # for _ in range(emphasize_experiments):
    #     all_experience.extend(experiment_experience())
    #
    # x = [i[0] for i in all_experience]
    # y = [i[1] for i in all_experience]

    ds_train = pd.read_csv(URL + "ds_train_resampled.csv", delimiter="ᶶ", engine='python')
    ds_val = pd.read_csv(URL + "ds_val.csv", delimiter="ᶶ", engine='python')
    ds_test = pd.read_csv(URL + "ds_test.csv", delimiter="ᶶ", engine='python')
    
    list_columns = ['triple', 'bgp', 'join', 'leftjoin', 'union', 'filter', 'project',
       'distinct', 'treesize', 'pcs0', 'pcs1', 'pcs2', 'pcs3', 'pcs4', 'pcs5',
       'pcs6', 'pcs7', 'pcs8', 'pcs9', 'pcs10', 'pcs11', 'pcs12', 'pcs13',
       'pcs14', 'pcs15', 'pcs16', 'pcs17', 'pcs18', 'pcs19', 'pcs20', 'pcs21',
       'pcs22', 'pcs23', 'pcs24']
    x_train_query = ds_train[list_columns]
    x_val_query   = ds_val[list_columns]
    x_test_query  = ds_test[list_columns]
    
    x_train_tree = ds_train['trees'].values
    x_val_tree = ds_val['trees'].values
    x_test_tree = ds_test['trees'].values
    
    y_train = ds_train['time'].values
    y_val = ds_val['time'].values
    y_test = ds_test['time'].values
    
    scalerx = StandardScaler()
    x_train_scaled = scalerx.fit_transform(x_train_query);
    x_val_scaled = scalerx.transform(x_val_query);
    x_test_scaled = scalerx.transform(x_test_query);
    
    #Scale x_query data.
    scaled_df_train = pd.DataFrame(x_train_scaled, index=x_train_query.index, columns=x_train_query.columns)
    scaled_df_val = pd.DataFrame(x_val_scaled, index=x_val_query.index, columns=x_val_query.columns)
    scaled_df_test = pd.DataFrame(x_test_scaled, index=x_test_query.index, columns=x_test_query.columns)
    
    
    reg = model.NeoRegression(have_cache_data=False, aec_file=aec_dir, epochs=200, verbose=verbose)
    
    #Fit the transformer tree data
    reg.fit_transform_tree_data(x_train_tree, x_val_tree, x_test_tree)
    
    #Fit model
    reg.fit(x_train_tree, scaled_df_train.values, y_train, x_val_tree, scaled_df_val.values, y_val)
    reg.save(fn)
    
    #Prediction
    preds_val = reg.predict(x_val_tree, scaled_df_val.values)
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    print("RMSE in VAL: {}".format(rmse))
    reg.scatter_image(preds_val, y_val,"Scatter real latency vs prediction on Val dataset.", "neo_with_aec_scatter_val")
    
    preds_test = reg.predict(x_test_tree, scaled_df_test.values)
    rmsetest = np.sqrt(mean_squared_error(y_test, preds_test))
    print("RMSE in TEST: {}".format(rmsetest))
    reg.scatter_image( preds_test, y_test, "Scatter real latency vs prediction on Test dataset.", "neo_with_aec_scatter_test")

    return reg

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: train.py MODEL_FILE")
        exit(-1)
    
    if (sys.argv[2] == 'bao'):
        train_and_save_model(sys.argv[1]+"_"+sys.argv[2])
    elif(sys.argv[2] == 'neo') :
        train_and_save_neo_model(sys.argv[1]+"_"+sys.argv[2], './nodesautoencoder_bao.pth')

    # print("Model saved, attempting load...")
    # reg = model.BaoRegression(have_cache_data=False)
    # reg.load(sys.argv[1])

