import datetime
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import data_preprocessing
from Models.model_trees_algebra import NeoRegression as NeoRegression
from Models.model_trees_algebra_aec import NeoRegression as AECNeoRegression
from Models.model import BaoRegression
import numpy as np
import pandas as pd

import argparse
import os.path as osp

# Increase the limit of the csv field size to avoid the error "field larger than field limit (131072)"
import sys
import csv
csv.field_size_limit(sys.maxsize)

def train_and_save_model_bao(
    ds_train, ds_val, ds_test, output_path, verbose=True
):
    x_train_query = ds_train[data_preprocessing.LIST_QUERY_COLUMNS]
    x_train_query_json_card = ds_train[data_preprocessing.CARDINALITY_COLUMNS]

    x_val_query = ds_val[data_preprocessing.LIST_QUERY_COLUMNS]
    x_val_query_json_card = ds_val[data_preprocessing.CARDINALITY_COLUMNS]

    x_test_query = ds_test[data_preprocessing.LIST_QUERY_COLUMNS]
    x_test_query_json_card = ds_test[data_preprocessing.CARDINALITY_COLUMNS]

    x_train_tree = ds_train["trees"].values
    x_val_tree = ds_val["trees"].values
    x_test_tree = ds_test["trees"].values

    y_train = ds_train["time"].values
    y_val = ds_val["time"].values
    y_test = ds_test["time"].values

    scalerx = StandardScaler()
    x_train_scaled = scalerx.fit_transform(x_train_query)
    x_val_scaled = scalerx.transform(x_val_query)
    x_test_scaled = scalerx.transform(x_test_query)

    # Scale x_query data.
    scaled_df_train = pd.DataFrame(
        x_train_scaled, index=x_train_query.index, columns=x_train_query.columns
    )
    x_train_query = data_preprocessing.concat_dataframes(
        scaled_df_train, x_train_query_json_card
    )

    scaled_df_val = pd.DataFrame(
        x_val_scaled, index=x_val_query.index, columns=x_val_query.columns
    )
    x_val_query = data_preprocessing.concat_dataframes(
        scaled_df_val, x_val_query_json_card
    )

    scaled_df_test = pd.DataFrame(
        x_test_scaled, index=x_test_query.index, columns=x_test_query.columns
    )
    x_test_query = data_preprocessing.concat_dataframes(
        scaled_df_test, x_test_query_json_card
    )

    max_cardinality = data_preprocessing.get_max_cardinaliy(x_train_query_json_card)

    verbose = True
    reg = BaoRegression(epochs=2, verbose=verbose, output_path=output_path)

    # Fit the transformer tree data
    reg.fit_transform_tree_data(ds_train, ds_val, ds_test)

    x_train_query["json_cardinality"] = x_train_query["json_cardinality"].apply(
        lambda x: data_preprocessing.pred2index_dict(x, reg.get_pred(), max_cardinality)
    )
    x_val_query["json_cardinality"] = x_val_query["json_cardinality"].apply(
        lambda x: data_preprocessing.pred2index_dict(x, reg.get_pred(), max_cardinality)
    )
    x_test_query["json_cardinality"] = x_test_query["json_cardinality"].apply(
        lambda x: data_preprocessing.pred2index_dict(x, reg.get_pred(), max_cardinality)
    )

    # Fit model
    reg.fit(
        x_train_tree,
        y_train,
        x_val_tree,
        y_val,
    )
    reg.save(osp.join(output_path, "regressor"))

    preds_test = reg.predict_custom(x_test_tree, y_test)
    rmsetest = np.sqrt(mean_squared_error(y_test, preds_test))
    print("RMSE in TEST: {}".format(rmsetest))
    reg.scatter_image(
        preds_test,
        y_test,
        "Scatter real latency vs prediction on Test dataset.",
        osp.join(output_path, "model_with_aec_scatter_test"),
    )

    # Save predictions on test dataset along with real values
    preds_test_df = pd.DataFrame({'Real Value': ds_test["time"].values, 'Predicted Value': preds_test})
    preds_test_df.to_csv(osp.join(output_path, 'predictions_test.csv'), index=False)

    # Save RMSE on test dataset to a file
    with open(osp.join(output_path, 'rmse_test.txt'), 'w') as f:
        f.write(f'{rmsetest}')

    return reg

def train_and_save_model(
    ds_train, ds_val, ds_test, output_path, aec=None, verbose=True
):

    x_train_query = ds_train[data_preprocessing.LIST_QUERY_COLUMNS]
    x_train_query_json_card = ds_train[data_preprocessing.CARDINALITY_COLUMNS]

    x_val_query = ds_val[data_preprocessing.LIST_QUERY_COLUMNS]
    x_val_query_json_card = ds_val[data_preprocessing.CARDINALITY_COLUMNS]

    x_test_query = ds_test[data_preprocessing.LIST_QUERY_COLUMNS]
    x_test_query_json_card = ds_test[data_preprocessing.CARDINALITY_COLUMNS]

    x_train_tree = ds_train["trees"].values
    x_val_tree = ds_val["trees"].values
    x_test_tree = ds_test["trees"].values

    y_train = ds_train["time"].values
    y_val = ds_val["time"].values
    y_test = ds_test["time"].values

    scalerx = StandardScaler()
    x_train_scaled = scalerx.fit_transform(x_train_query)
    x_val_scaled = scalerx.transform(x_val_query)
    x_test_scaled = scalerx.transform(x_test_query)

    # Scale x_query data.
    scaled_df_train = pd.DataFrame(
        x_train_scaled, index=x_train_query.index, columns=x_train_query.columns
    )
    x_train_query = data_preprocessing.concat_dataframes(
        scaled_df_train, x_train_query_json_card
    )

    scaled_df_val = pd.DataFrame(
        x_val_scaled, index=x_val_query.index, columns=x_val_query.columns
    )
    x_val_query = data_preprocessing.concat_dataframes(
        scaled_df_val, x_val_query_json_card
    )

    scaled_df_test = pd.DataFrame(
        x_test_scaled, index=x_test_query.index, columns=x_test_query.columns
    )
    x_test_query = data_preprocessing.concat_dataframes(
        scaled_df_test, x_test_query_json_card
    )

    max_cardinality = data_preprocessing.get_max_cardinaliy(x_train_query_json_card)

    verbose = True
    if aec:
        reg = AECNeoRegression(
            epochs=2, verbose=verbose, output_path=output_path, aec=aec
        )
    else:
        reg = NeoRegression(epochs=100, verbose=verbose, output_path=output_path)

    # Fit the transformer tree data
    reg.fit_transform_tree_data(ds_train, ds_val, ds_test)

    x_train_query["json_cardinality"] = x_train_query["json_cardinality"].apply(
        lambda x: data_preprocessing.pred2index_dict(x, reg.get_pred(), max_cardinality)
    )
    x_val_query["json_cardinality"] = x_val_query["json_cardinality"].apply(
        lambda x: data_preprocessing.pred2index_dict(x, reg.get_pred(), max_cardinality)
    )
    x_test_query["json_cardinality"] = x_test_query["json_cardinality"].apply(
        lambda x: data_preprocessing.pred2index_dict(x, reg.get_pred(), max_cardinality)
    )

    # Fit model
    reg.fit(
        x_train_tree,
        x_train_query.values,
        y_train,
        x_val_tree,
        x_val_query.values,
        y_val,
    )
    reg.save(osp.join(output_path, "regressor"))

    # Prediction
    _, _, y_val = reg.json_loads(
        x_val_tree, x_val_query.values, y_val
    )
    preds_val = reg.predict_raw_data(x_val_tree, x_val_query.values)
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    print("RMSE in VAL: {}".format(rmse))
    reg.scatter_image(
        preds_val,
        y_val,
        "Scatter real latency vs prediction on Val dataset.",
        osp.join(output_path, "model_with_aec_scatter_val"),
    )

    _, _, y_test = reg.json_loads(
        x_test_tree, x_test_query.values, y_test
    )
    preds_test = reg.predict_raw_data(x_test_tree, x_test_query.values)
    rmsetest = np.sqrt(mean_squared_error(y_test, preds_test))
    print("RMSE in TEST: {}".format(rmsetest))
    reg.scatter_image(
        preds_test,
        y_test,
        "Scatter real latency vs prediction on Test dataset.",
        osp.join(output_path, "model_with_aec_scatter_test"),
    )

    # Save predictions on test dataset along with real values
    preds_test_df = pd.DataFrame({'Real Value': ds_test["time"].values, 'Predicted Value': preds_test})
    preds_test_df.to_csv(osp.join(output_path, 'predictions_test.csv'), index=False)

    # Save RMSE on test dataset to a file
    with open(osp.join(output_path, 'rmse_test.txt'), 'w') as f:
        f.write(f'{rmsetest}')

    return reg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create training data for embedding model"
    )

    parser.add_argument(
        "--data-dir", dest="data_dir", help="Where is the data stored", required=True
    )

    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Where to store model outputs",
        required=True,
    )

    parser.add_argument(
        "--seed",
        dest="seed",
        help="seed to split train, val data",
        default=0,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--val-rate",
        dest="val_rate",
        help="rate to split and use as val dataset",
        default=0.2,
        type=float,
        required=False,
    )

    parser.add_argument(
        "--data-source",
        dest="data_source",
        help="Could be 'kaggle' or 'huggingface' dataset",
    )
    parser.add_argument(
        "--verbose", dest="verbose", help="", type=bool, default=True, required=False
    )
    parser.add_argument(
        "--with-aec", dest="with_aec", help="", type=bool, default=False, required=False
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    ds_train, ds_val, ds_test = data_preprocessing.prepare_datasets(
        args.data_dir, val_rate=args.val_rate, seed=args.seed, model_id=model_id
    )
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    output = osp.join(args.output_dir, model_id)
    if not os.path.isdir(output):
        os.mkdir(output)
    aec = (
        {
            "train_aec": True,
            "aec_file": osp.join(output, "aec_model.pth"),
            "aec_epochs": 10,
        }
        if args.with_aec
        else None
    )
    #train_and_save_model(ds_train, ds_val, ds_test, output, aec=aec)
    train_and_save_model_bao(ds_train, ds_val, ds_test, output + "/")
