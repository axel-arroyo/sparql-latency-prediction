import datetime
import logging
import os

from sklearn.model_selection import train_test_split
import pandas as pd
import os.path as osp
import json

ADDITIONAL_QUERY_COLUMNS = [
    'projectVarCount', 'bgpCount', 'joinVertexCount', 'tpCount'
]

LIST_QUERY_COLUMNS = [
    "filter_bound",
    "filter_contains",
    "filter_eq",
    "filter_exists",
    "filter_ge",
    "filter_gt",
    "filter_isBlank",
    "filter_isIRI",
    "filter_isLiteral",
    "filter_lang",
    "filter_langMatches",
    "filter_le",
    "filter_lt",
    "filter_ne",
    "filter_not",
    "filter_notexists",
    "filter_or",
    "filter_regex",
    "filter_sameTerm",
    "filter_str",
    "filter_strends",
    "filter_strstarts",
    "filter_subtract",
    "has_slice",
    "max_slice_limit",
    "max_slice_start",
]

CARDINALITY_COLUMNS = ["json_cardinality"]


def create_df_from_data(data, index, columns):
    return pd.DataFrame(data, index=index, columns=columns)


def concat_dataframes(df1, df2):
    return pd.concat([df1, df2], axis=1)


def split_train_data(all_data: pd.DataFrame, val_rate: float, seed: int):

    ranges = {
        "1_2": all_data[(all_data["time"] > 0) & (all_data["time"] <= 2)],
        "2_3": all_data[(all_data["time"] > 2) & (all_data["time"] <= 3)],
        "3_4": all_data[(all_data["time"] > 3) & (all_data["time"] <= 4)],
        "4_5": all_data[(all_data["time"] > 4) & (all_data["time"] <= 5)],
        "5_8": all_data[(all_data["time"] > 5) & (all_data["time"] <= 8)],
        "8_10": all_data[(all_data["time"] > 8) & (all_data["time"] <= 10)],
        "10_20": all_data[(all_data["time"] > 10) & (all_data["time"] <= 20)],
        "20_30": all_data[(all_data["time"] > 20) & (all_data["time"] <= 30)],
        "30_40": all_data[(all_data["time"] > 30) & (all_data["time"] <= 40)],
        "40_50": all_data[(all_data["time"] > 40) & (all_data["time"] <= 50)],
        "50_60": all_data[(all_data["time"] > 50) & (all_data["time"] <= 60)],
        "60_80": all_data[(all_data["time"] > 60) & (all_data["time"] <= 80)],
        "80_100": all_data[(all_data["time"] > 80) & (all_data["time"] <= 100)],
        "100_150": all_data[(all_data["time"] > 100) & (all_data["time"] <= 150)],
        "150_last": all_data[(all_data["time"] > 150)],
    }
    train_data = []
    val_data = []

    for rang in ranges.values():
        if rang.shape[0] >= 3:
            X_train, X_val = train_test_split(
                rang, test_size=val_rate, shuffle=True, random_state=seed
            )

            train_data.append(X_train)
            val_data.append(X_val)
    train_data_list = pd.concat(train_data)
    val_data_list = pd.concat(val_data)
    print(
        "Shapes : Train: {} Val: {}".format(train_data_list.shape, val_data_list.shape)
    )
    return train_data_list, val_data_list


def getmax(x):
    lista = list(x.values())
    maximo = 0
    for el in lista:
        if maximo < float(el):
            maximo = float(el)
    return maximo


def get_max_cardinaliy(x_train_query):
    return (
        x_train_query["json_cardinality"]
        .apply(lambda x: json.loads(x))
        .apply(lambda x: getmax(x))
        .max()
    )


def prepare_datasets(
    data_dir,
    val_rate,
    seed,
    model_id=datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
    ds_test_file_name="ds_test.csv",
    ds_train_file_name="ds_train_val.csv",
):
    ds_test = pd.read_csv(
        osp.join(data_dir, ds_test_file_name), delimiter="ᶶ", engine="python"
    )
    data_train_val = pd.read_csv(
        osp.join(data_dir, ds_train_file_name), delimiter="ᶶ", engine="python"
    )

    print("Shape: train_data", data_train_val.shape)

    query_exec_time_threshold = 65  # use queries under 65 seconds
    data_train_val = data_train_val[data_train_val["time"] <= query_exec_time_threshold]
    ds_test = ds_test[ds_test["time"] <= query_exec_time_threshold]
    ds_train, ds_val = split_train_data(data_train_val, val_rate=val_rate, seed=seed)

    if not os.path.isdir(osp.join(data_dir, model_id)):
        os.mkdir(osp.join(data_dir, model_id))

    train_path = osp.join(data_dir, model_id, "ds_train.csv")
    val_path = osp.join(data_dir, model_id, "ds_val.csv")
    test_path = osp.join(data_dir, model_id, "ds_test.csv")

    ds_train.to_csv(train_path, index=False, sep="ᶶ")
    ds_val.to_csv(val_path, index=False, sep="ᶶ")
    ds_test.to_csv(test_path, index=False, sep="ᶶ")
    logging.info(f"Datasets created: {train_path}, {val_path}, {test_path}")
    return ds_train, ds_val, ds_test


def pred2index_dict(x, pred_to_index, max_cardinality):
    resp = {}
    x = json.loads(x)
    for el in x.keys():
        if el in pred_to_index:
            resp[pred_to_index[el]] = float(x[el]) / max_cardinality
    return resp
