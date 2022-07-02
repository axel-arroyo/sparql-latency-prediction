## SPARQL query execution time prediction using Deep Learning companion Git repository

### Usage

To reproduce the experiments download the dataset from:
- kaggle:
```
kaggle datasets download -d danielcasals/sparql-queries-performance-prediction
unzip sparql-queries-performance-prediction.zip
mv sparql-queries-performance-prediction/ds_test_pred_filtered.csv sparql-queries-performance-prediction/ds_test.csv               
mv sparql-queries-performance-prediction/ds_trainval_pred_filtered.csv  sparql-queries-performance-prediction/ds_train_val.csv
```
- huggingface datasets:
```
git lfs install
git clone https://huggingface.co/datasets/dacasals/sparql-wikidata-queries
```
Run the train script with:
```
usage: train.py [-h] --data-dir DATA_DIR --output-dir OUTPUT_DIR [--seed SEED] [--val-rate VAL_RATE] [--data-source DATA_SOURCE] [--verbose VERBOSE] [--with-aec WITH_AEC]
```
Jupyter Notebook ``ModelTreeConvSparql.ipynb`` trains and evaluates the model proposed in our work using the test set data. This first divides training data, and cleans and prepares the data.

Class ```Regression``` in [model_trees_algebra.py](model_trees_algebra.py), has the functions for preparing data and training and evaluating the model we propose.
Our model's architecture is in [NeoNet](net.py), which is composed by query layers and tree convolution laters.

The TreeConvolution folder has the Tree Convolution Neural Network (TCNN) implementation, which we imported from [@github.com:learnedsystems/BaoForPostgreSQL/TreeConvolution](https://github.com/learnedsystems/BaoForPostgreSQL/tree/master/bao_server/TreeConvolution).
The Python file ```tcnn.py``` implements:  
 - ``BinaryTreeConv``: TCNN convolution layer implementation using ``nn.Conv1D``.
 - ``TreeActivation``: Activation function.
 - ``TreeLayerNorm``: Normalization layer implementation, appied after ``BinaryTreeConv``.
 - ``DynamicPooling``: Dynamic pooling for converting the values from the last convolution layer into a fix length vector. 
 - ``BinaryTreeConvWithQData``: Our query level characteristics implementation, which we concatenate with query plan characteristics. We use this as our first TCNN layer. See [BinaryTreeConvWithQData.py](TreeConvolution/tcnn.py)
 
### Requirements.
We used an AMD opteron server, using 64GB of RAM memory and a Nvidia 2080ti GPU in the training, testing and validation process. 
We implemented our network using ``pytorch`` using 3rd party libraries such as: ``pandas``,``numpy``,``plotly``,``matplotlib``, ``sklearn``.
