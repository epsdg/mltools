# mltools

mltools is a packaging of standard machine learning models from [scikit-learn](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning), [XGBoost](https://xgboost.readthedocs.io), [LightGBM](https://lightgbm.readthedocs.io),  [TensorFlow](https://www.tensorflow.org/guide/summaries_and_tensorboard), and the tf.keras implementation of [Keras](https://keras.io/).

mltools features include:
* Consistent interface for common ML libraries
* Simplified access to hyperparameters via a yaml file
* Detailed, persistent experiment logging.  Hyperparameters, features, and cross-validation metrics of all experiments are logged to stdout and to a log file.  The same logger instance can also be used for feature importance results and other relevant info.  TensorFlow and Keras models also generate output to Tensorboard summaries.
* Efficient CV for model stacking, using a persistent index of fold assignments to prevent data leakage across folds.
* Parallel/multiprocessing CV workflows using your choice of backend ([concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html), [joblib.Parallel](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html), [dask](https://docs.dask.org/en/latest/), or [dask-distributed](http://distributed.dask.org/en/latest/index.html)).  Multithreading / multiprocessing is the default implementation for scikit-learn model sets.  Use dask-distributed for convenient deployment on clusters or multi-core instances, with a convenient [dashboard](http://distributed.dask.org/en/latest/web.html).    *XGBoost, LightGBM, TensorFlow, and Keras models rely on the default multiprocessing implementation.*


## Model Sets

### scikit-learn
**models.skl_model.SKLModelSet**: Supports most classifiers from the [scikit-learn](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning) library, including:
* Logistic regression
* Support vector classifier (SVC) using SGD
* SGD classifier
* Random forests
* AdaBoost
* k-nearest neighbors (knn)

### MART/gradient boosted machine

**models.gbm_model.GBMModel**: MART/boosted trees using either
* [XGBoost](https://xgboost.readthedocs.io) (`library: 'xgb'`), or
* [LightGBM](https://lightgbm.readthedocs.io) (`library: 'lgbm'`).

Allows for easy switching between the two libraries using the same dataset.  Supports classification or regression depending on objective set in the params file.  **Note:** params are different depending on the library selected.  Consult the API ref for [XGBoost](https://xgboost.readthedocs.io/en/latest/parameter.html) or [LightGBM](https://lightgbm.readthedocs.io/en/latest/Parameters.html) for specific parameter options.  A sample params file for each library is provided in `/mltools/sample_params`.

### Tensorflow
**models.tf_ffn.FFNClassifier**: [TensorFlow](https://www.tensorflow.org/guide/summaries_and_tensorboard) implementation of a feedforward multi-layer perceptron (MLP) classifier using a cross entropy loss.  Currently supports binary classification only.  Requires TensorFlow 1.13.  *Will not be updated to support TensorFlow 2.x (use keras_ffn models instead)*

Config options include:
* Customize the number of layers and units per layer
* Norm regularization: l1/sparse, l2, and blended l1/l2
* [Dropout](http://jmlr.org/papers/v15/srivastava14a.html) on inputs and/or hidden layers)
* Optimizers: supports stochastic graidient descent, AdaGrad, AdaDelta, and Adam
* Activation functions: ReLU, leaky ReLU, ELU, tanh, sigmoid, and others
* [Batch normalization](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)
* Positive-class weighting factor for unbalanced classes in the training data.
* Metrics via [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard), as well as detailed log results

**models.tf_ffn.FFNNRegressor**:  Regression with the same config options as DNNClassifier.

### Keras
**models.keras_ffn.FFNBinaryClassifier**:  tf.keras implementation of the feedforward classifier.  Supports binary classification only.  Requires TensorFlow 1.13, and will be updated to support TensorFlow 2.0.  Supports all regularization/optimization options listed above for tf_ffn, plus several additional methods and config enhancements (detailed below)

**models.keras_ffn.FFNRegressor**:  tf.keras implementation of the feedforward classifier.  Requires TensorFlow 1.13, and will be updated to support TensorFlow 2.0.  Supports all regularization/optimization options listed above for tf_ffn, plus some additional methods and config enhancements (detailed below)


## Basic usage
##### Create a model instance

New instances of all models require X_train, y_train, X_test, and a path to params_file.  X_train and X_test are pandas DataFrames.  y_test and folds_lookup are pandas Series.  X_train, y_train, X_test, and folds_lookup must have an unduplicated, single-tier integer index so that prediction outputs can be merged with the inputs.  X_train, y_train, and folds_lookup should have matching indices.


**Args**:
* `X_train`: pandas DataFrame, unduplicated integer index
* `y_train`: pandas seris, integer index matching X_test
* `X_test`: pandas DataFrame, unduplicated integer index
* `params_file`, string, path/filename for params.yaml file
* `folds_lookup`: pandas Series or int, optional:
  * *if pandas Series*, values are integer fold assignments numbered 1-n_folds; int index matches X_train and y_train indices
  * *if int*, `folds_lookup` random folds will be generated
  * *if None*, 5 random folds will be generated
* `prefix`: string, optional, prepended to log files and predictions output columns.  Defaults to 'mltools'
* `weights`: 1-d array or list of floats, optional, instance weights for train set
* `tf_path` (*tf/keras models only*), string, path/filename for tensorboard logs and checkpoint files.  Defaults to `/mltools/tf`.
* `library` (*MART/GBM models only*) string in {'xgb', 'lgbm'}: MART/GBM backend library,  Defaults to Xgboost.
* `backend` (*scikit-learn models only*) string, optional, parallel backend to use for cross-validation routines:
  * `'mp'`: multiprocessing.pool
  * `'tpe'`: concurrent.futures.ThreadPoolExecutor
  * `'ppe'`: concurrent.futures.ProcessPoolExecutor
  * `'joblib-threads'`: joblib.Parallel(prefer='threads')
  * `'joblib-processes'`: joblib.Parallel(prefer='processes')
  * `'dask'`: dask
  * `'dask-dist'`: dask distributed
* `n_processes` (*scikit-learn models only*) int, optional, number of backend threads or processes
* `logger`, one of:
  * python logging.Logger instance
  * string, prefix for log filename
  * None to generate default a default logger instance 'mltools'

```python
from mltools.utils import get_logger
from mltools.models.keras_ffn import FFNRegressor

logger = utils.get_logger('mltools_example')

model = FFNRegressor(X_train, y_train, X_test,
                     folds_lookup,
                     'params.yaml',
                     weights=None
                     prefix='ml_test',
                     tf_path=r'~/data/tf',
                     logger=logger)
```
##### Load parameters

Load params from the params_file path/filename
```python
model.load_hparams()
```

##### Preprocess inputs

* Available on all model classes.
* Recommend mean-filling for all models unless you're sure there are no missing values.
* Recommend standardizing for all tf/keras models and most sklearn models (all linear model subclasses)

  **Args**:
  * `fillna`, bool, fill NaNs
  * `fill_with`, string in {'mean', 'median'}
  * `standardize`, bool, center and standard-scale inputs
  * `clip_outliers`, int or None.  If int:
    * clip all input values > `clip_outliers` to `clip_outliers`
    * clip all input values < **-**`clip_outliers` to **-**`clip_outliers`

```python
model.preprocess(fillna=True,
                 fill_with='mean',
                 standardize=True,
                 clip_outliers=None,
                 use_PCA=True,
                 n_PCA_components=50))
```

For skl_model.SKLModelSet, the preprocess method also generates a PCA dataset (`use_PCA=True`).  Pass `n_PCA_components` to limit PCA inputs to the n primary principal components To include all PCA components in the dataset, set `n_PCA_components=-1`

*PCA is not implemented in GBM and tf/keras models.*

##### Grid CV

Set grid values for cross-validation:
```python
model.cv_grid = {'layers': [[512, 512, 512], [1024, 512, 256, 128]],
                'activation': ['relu', 'elu']}
```

Perform cross validation, with all results are sent to the logger:
```python
model.grid_cv(val_rounds=2)
```

Results of each cross-validation run are automatically sent to the logger.  To access the model's results (as a pd.DataFrame):
```python
results_df = model.cv_results
```


##### Generate predictions
CV predictions for each CV fold:
```python
cv_outputs = model.cv_outputs()
```
Test set predictions, trained on the full training set:
```python
test_outputs = model.test_outputs()
```
Predictions are returned as pandas DataFrame.

##### Plot training metrics

Implemented for GBMModel and tf/keras FFN models only.  Plots round-by-round validation metrics (depending on the model and metrics selected) of the last completed train/eval routine.  Plots displayed inline in notebooks, and also saved as `./evals_plot.png`.
```python
model.plot_results(filename='evals_plot.png')
```

##### Plot regression predictions

Implemented for GBM and tf/keras.  When calling `grid_cv()`, pass `plot_n_samples` (sample predictions per fold) to generate a plot of predicted vs. actual regression results by fold.  Plots displayed inline in notebooks, and also saved as `./CV_preds_plot.png`.
```python
model.grid_cv(val_rounds=5, plot_n_samples=None)
```

## Sample params file
Sample params for each model type are provided in `/mltools/sample_params`.  Many params are optional, but recommend inclulding all params in the sample files.
```yaml
train_mode: minibatch_gd  # sgd, minibatch_gd
epoch_train_batches: 5  # sgd only, ignored for minibatch_gd
epoch_val_batches: 5  # sgd only, ignored for minibatch_gd

n_epochs: 48

loss_fn: mean_squared_error
# mean_squared_error, mean_absolute_error, binary_crossentropy, ...

optimizer: adam
train_batch_size: 500
eta: 0.003
decay: 0.1
beta_1: 0.9
beta_2: 0.999
epsilon: 1e-8
momentum: 0.999
amsgrad: False  # for optimizer='adam' only

reduce_eta: False  # do not use with find_eta or use_clr
reduce_eta_steps: 5
reduce_eta_factor: 0.2
early_stop_rounds: 0

find_eta: False # do not use with reduce_eta
find_eta_start: 1e-06 # recommend 1e-07 - 1e-06
find_eta_factor: 1.2  # recommend ~ 1.1 (fine-tuning) - 1.4 (fast)

use_clr: True  # do not use with find_eta or reduce_eta
clr_min: 0.0003
clr_max: 0.01
clr_period: 8

val_batch_size: 500
verbose: 0
use_tensorboard: True

init_mode: fan_avg    #  fan_in, fan_out, fan_avg
init_distribution: normal
layers: [512, 512, 512, 512, 512]
activation: elu   #  relu, elu, selu, crelu, tanh
regularizer: l1-l2   # l1, l2, l1-l2, None
drop_rates: [0.2, 0.2, 0.2, 0.2, 0.2]
l2_reg_weight: 5e-7
l1_reg_weight: 2e-7
use_batch_norm: [True, True, True, True, True]
batch_norm_momentum: 0.9999
pos_weight: 1

predict_chunk_size: 1000
binary_metrics: tf  # tf, sklearn, both
```

Sample log results:
```
10:31:58: filling NaN...
10:32:04: standardizing inputs...
10:32:09: preprocessing complete.
10:32:09: base params:
          -  train_mode: minibatch_gd
          -  epoch_train_batches: 5
          -  epoch_val_batches: 5
          -  n_epochs: 48
          -  loss_fn: mean_squared_error
          -  optimizer: sgd
          -  train_batch_size: 500
          -  eta: 0.003
...

10:32:09: ------------------------ FOLD 1 OF 5 ------------------------
10:32:10: clr_min=0.0003  
10:32:12: training 800 iterations per epoch
10:32:13:                  TRAIN       |               VAL        |
10:32:13: EPOCH       MSE         MAE  |        MSE          MAE  |     eta
10:32:30:    1    35.5296      4.8536  |  4845.7500    1144.9674  |  3.0e-04
10:32:46:    2    24.4284      4.0025  |  2406.1562     453.1542  |  1.0e-02
10:33:01:    3     8.7243      2.3302  |  1322.4580      83.5650  |  2.0e-02
10:33:16:    4     9.0259      2.3763  |   739.8892      66.9962  |  3.0e-02
10:33:32:    5     7.6834      2.1746  |   554.0875      19.9541  |  4.0e-02
10:33:47:    6     7.2638      2.1065  |   370.1605      15.7916  |  3.0e-02
10:34:02:    7     6.9415      2.0625  |   104.8456       6.9945  |  2.0e-02
...
...
11:33:52:    best val MSE: 8.2830, round 44
11:33:52:    best val MAE: 2.2673, round 46
11:33:52:
11:33:52: grid CV complete.
11:33:53: CV results summary, validation scores:
clr_min           best_MSE     rnd_MSE    best_MAE     rnd_MAE
0.0003              7.1234     39.0000      2.0618     36.8000
0.001               7.2158     34.0000      2.3266     35.4000
```

## Model-specific features & methods:

### Keras models:

Keras models support all activation functions and optimizers implemented in tensorflow.keras (tf 1.13).

Keras models implement the several optimization enhancements, accessed through the params file:
* **Learning rate finder**: based on [[1]](https://arxiv.org/abs/1506.01186), [[2]](https://arxiv.org/abs/1708.07120); but using an exponential step instead of a linear step.  The step is calculated as:

  `eta = find_eta_start * (find_eta_factor ** epoch)`, epoch ∈ {0, ..., n_epochs-1}

  The learning rate finder starts with a small eta (learning rate), and increases eta each epoch.  A starting eta of ~1e-07, combined with a factor of ~1.2 generally yields an optimal eta range within ~50 epochs.
* **Cyclical learning rate (CLR)**, based on [[1]](https://arxiv.org/abs/1506.01186), [[2]](https://arxiv.org/abs/1708.07120); using a triangular cycle/window

* **Reduce eta on training plateau**, using the Keras ReduceLROnPlateau callback.  Monitors plateau of training loss.
* **Early stopping**, monitors plateau of validation loss.

#### tf.keras binary metrics:

`FFNBinaryClassifier` implements binary metrics (accuracy, precision, recall, and ROC AUC) from both scikit-learn.metrics and tf.metrics (tf.keras.metrics in tf 2.0).  Some metric values from sklearn and tf/tf.keras may differ by ~1-2%.

General recommendations:
* For faster training, set `binary_metrics: 'tf'`
* For faster training, set `binary_metrics: 'sklearn'`
* For faster training, set `binary_metrics: 'both'`

For a detailed explanation, see the footnote below.


### scikit-learn models:

The `cv_predictions` and `test_predictions` methods implement multiprocessing (with user choice of backend) by default.  For a (usually much slower) loop-based version, use `cv_predictions_loop` and `test_predictions_loop`.

---
### A note on keras_ffn binary metrics:
Training set metrics: tf.keras.metrics objects update the training metric batch-by-batch, instead of by epoch.  For train set metrics, the 'tf' version of precision and recall reflect precision and recall on the final training batch, *not* the full training set.  The 'skl' version reflects train precision/recall on the full train set.

Validation metrics: accuracy, precision and recall should be the same for 'tf' and 'skl' because both use the full validation set.  ROC AUC values differ as detailed below.

To select which metrics are displayed in the logs, use the 'binary_metrics' param.


#### summary of metric outputs:
| metric  | binary_metrics='keras' | binary_metrics='sklearn' | binary_metrics='both'
| - | - | - | - |
| logs: accuracy  | tf.keras.metrics.Precision()  | calculated from sklearn.metrics.confusion_matrix()| both
| logs: precision  | tf.keras.metrics.Precision()  | calculated from sklearn.metrics.confusion_matrix() | both
| logs: recall  | tf.keras.metrics.Recall()  | calculated from sklearn.metrics.confusion_matrix() | both
| logs: ROC AUC  | tf.metrics.AUC  | calculated from sklearn.metrics.roc_curve and sklearn.metrics.auc | both
| Tensorboard  | accuracy, precision, recall, ROC AUC, binary logloss, using tf.keras metrics | logloss only | accuracy, precision, recall, AUC, binary logloss, using tf.keras metrics
| CV metrics summary  | tf/keras metrics   | scikit-learn metrics | scikit-learn metrics

#### summary of discrepancies between metrics:
|metric | train | validation|
| - | - | - |
|accuracy | sklearn reflects full train set, tf/keras reflects final batch | same|
|precision | sklearn reflects full train set, tf/keras reflects final batch | same|
|recall | sklearn reflects full train set, tf/keras reflects final batch | same|
|AUC | different calculations *and* batch/epoch discrepancy | different calculations |
