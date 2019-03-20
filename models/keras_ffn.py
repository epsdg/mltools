import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, Dropout, BatchNormalization
from tensorflow.keras import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
import gc
from .. import utils
from ..config import cfg
from .base_model import BaseModel


class FFNModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, params_file=None, folds_lookup=None,
                 prefix=None, weights=None, tf_path=None, logger=None):
        BaseModel.__init__(self, X_train, y_train, X_test, params_file, folds_lookup,
            prefix, logger)

        self.model = None
        self.sess = None
        self.history = None

        self.weights = weights
        self.n_inputs = None
        self.initializer = None
        self.regularizer = None
        self.activation = None

        self.tf_path = tf_path
        self.logdir = None
        self.output_suffix = '_keras_pred'


    def init_hparams(self):
        '''interpret params.yaml file to set tf.Graph params
        '''
        self.n_inputs = self.X_train.shape[1]

        self.initializer = tf.keras.initializers.VarianceScaling(
                                scale=1.0,
                                mode=self.params['init_mode'],
                                distribution=self.params['init_distribution'],
                                seed=None)

        l1_reg=float(self.params.get('l1_reg_weight', 0.0))
        l2_reg=float(self.params.get('l2_reg_weight', 0.0))

        reg={'None': None,
             'l1': tf.keras.regularizers.l1(l1_reg),
             'l2': tf.keras.regularizers.l2(l2_reg),
             'l1-l2': tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)}

        self.regularizer = reg.get(self.params['regularizer'], None)

        eta = float(self.params.get('eta', 0.001))
        momentum=float(self.params.get('momentum', 0.0))
        beta_1=float(self.params.get('beta1', 0.9))
        beta_2=float(self.params.get('beta2', 0.999))
        epsilon=float(self.params.get('epsilon', 1e-08))
        decay=float(self.params.get('decay', 0.0))
        amsgrad=self.params.get('amsgrad', False)

        optimizers = {
            'sgd': tf.keras.optimizers.SGD(lr=eta,
                                       momentum=momentum,
                                       decay=decay),
            'adam': tf.keras.optimizers.Adam(lr=eta,
                                         beta_1=beta_1,
                                         beta_2=beta_2,
                                         epsilon=epsilon,
                                         decay=decay,
                                         amsgrad=amsgrad),
            'adagrad': tf.keras.optimizers.Adagrad(lr=eta,
                                               epsilon=epsilon,
                                               decay=decay),
            'adadelta': tf.keras.optimizers.Adadelta(lr=eta,
                                               epsilon=epsilon,
                                               decay=decay)}

        self.optimizer = optimizers[self.params.get('optimizer', 'sgd')]


    def get_sample_weights(self, fold):
        idx, _ = self._get_fold_indices(fold)
        return self.weights[idx.values.astype(int)]


    def init_tensorboard(self, fold='', param_set=''):
        '''Set directory and filename for tensorboard logs and checkpoint file
        '''
        now = datetime.now().strftime("%m%d-%H%M")
        comment = ''
        self.logdir = f'{self.tf_path}/tensorboard_logs/{now}-{fold}-{param_set}{comment}/'
        self.ckpt_file = f'{self.tf_path}/sessions/mlp.ckpt'


    def feedforward_layers(self, final_activation=None):
        '''Iterate layers of dropout-dense-batch norm'''
        X = Input(shape=(self.X_train.shape[1], ))

        layer = Layer(name='identity')(X)

        n_layers = len(self.params['layers'])

        for i, units in enumerate(self.params['layers']):

            drop_rate = self.params.get('drop_rates', [0.0] * n_layers)[i]
            if drop_rate > 0.0:
                layer = Dropout(drop_rate,
                                noise_shape=None,
                                seed=None,
                                name='drop_' + str(i+1))(layer)

            layer = Dense(units,
                        activation=self.params.get('activation', None),
                        kernel_initializer=self.initializer,
                        bias_initializer='zeros',
                        kernel_regularizer=self.regularizer,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        name='dense_' + str(i+1))(layer)

            if self.params.get('use_batch_norm', [False] * n_layers)[i]:
                layer = BatchNormalization(axis=-1,
                                           momentum=self.params.get('batch_norm_momentum', 0.99),
                                           epsilon=0.001,
                                           center=True,
                                           scale=True,
                                           beta_initializer='zeros',
                                           gamma_initializer='ones',
                                           moving_mean_initializer='zeros',
                                           moving_variance_initializer='ones',
                                           beta_regularizer=None,
                                           gamma_regularizer=None,
                                           beta_constraint=None,
                                           gamma_constraint=None,
                                           name='bn_'+str(i+1))(layer)

        outputs = Dense(1,
                    activation=final_activation,
                    kernel_initializer=self.initializer,
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None,
                    name='outputs')(layer)

        return X, outputs


    def find_lr(self, epoch, lr):
        '''Learning rate (eta) finder'''
        factor = float(self.params.get('find_eta_factor', 1.3))
        start_eta = float(self.params.get('find_eta_start', 1e-08))
        return factor ** (epoch) * start_eta


    def clr(self, epoch, lr):
        '''Cyclical learning rate'''
        eta_min = float(self.params.get('clr_min', 1e-05))
        eta_max = float(self.params.get('clr_max', 1e-02))
        clr_period = self.params.get('clr_period', 15)
        assert clr_period >= 4
        assert eta_min < eta_max
        if int(clr_period) % 2 > 0:
            clr_period = int(clr_period + 1)
        eta_range = eta_max - eta_min
        step = epoch % clr_period
        factor = abs( (clr_period // 2) - step) / (clr_period // 2)
        return eta_max - eta_range * factor


    def train_eval(self, X_train, y_train, X_val, y_val=None, weights=None,
                   return_preds=False, save_ckpt=False, fold=None, plot_n_samples=None):
        '''Core training and evals routine.
        Args:
            X_train, y_train, X_val, y_val: train and validation datasets
            weights: per-instance weighting factor for each instance in X_train
            return_preds: bool, False for train/eval, True to generate predictions (without evals)
            save_ckpt: bool, not implemented
            fold: integer fold number
            plot_n_samples: integer, number of samples for plot_regression_preds
        Returns: if return_preds, returns model predictions for X_val
        '''

        self.build_model(enable_metrics=not return_preds)

        callbacks = []

        if self.params.get('use_clr', False):
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(
                                                self.clr,
                                                verbose=0))

        if (self.params.get('reduce_eta', False)
            and not self.params.get('find_eta', False)
            and not self.params.get('use_clr', False)):
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                                                monitor='loss',
                                                factor=self.params['reduce_eta_factor'],
                                                patience=self.params['reduce_eta_steps'],
                                                verbose=1,
                                                mode='auto',
                                                min_delta=0.0001,
                                                cooldown=0,
                                                min_lr=1e-08))

        if not return_preds:
            # metrics callbacks, disabled for predictions-only mode
            epoch_begin = lambda epoch, logs: self.epoch_begin()

            epoch_end = lambda epoch, logs: self.epoch_end(
                    epoch, logs, X_train, y_train, X_val, y_val)

            train_end = lambda logs: self.train_end(
                    X_val, y_val, plot_n_samples, fold)

            if self.params.get('find_eta', False):
                callbacks.append(tf.keras.callbacks.LearningRateScheduler(
                                                    self.find_lr,
                                                    verbose=0))

            if self.params.get('use_tensorboard', False):
                callbacks.append(tf.keras.callbacks.TensorBoard(
                                        log_dir=self.logdir,
                                        histogram_freq=0,
                                        batch_size=self.params['val_batch_size'],
                                        write_graph=True,
                                        write_grads=False,
                                        write_images=False,
                                        update_freq='epoch'))

            if self.params.get('early_stop_rounds', False):
                callbacks.append(tf.keras.callbacks.EarlyStopping(
                                                    monitor='val_loss',
                                                    min_delta=0,
                                                    patience=self.params['early_stop_rounds'],
                                                    verbose=1,
                                                    mode='auto',
                                                    baseline=None,
                                                    restore_best_weights=False))

            callbacks.append(tf.keras.callbacks.LambdaCallback(
                                        on_epoch_begin=self.epoch_begin,
                                        on_epoch_end=epoch_end,
                                        on_batch_begin=None,
                                        on_batch_end=None,
                                        on_train_begin=self.train_begin,
                                        on_train_end=train_end))

            validation_data = (X_val, y_val)

        elif return_preds:
            validation_data=None

        train_batch_size = self.params.get('train_batch_size', 32)

        mode = self.params.get('train_mode', 'minibatch_gd')
        if mode == 'minibatch_gd':
            steps_per_epoch = None
            validation_steps = None
            n_batches = X_train.shape[0] // train_batch_size
            self.logger.info(f'training {n_batches} iterations per epoch')
        elif mode == 'sgd':
            steps_per_epoch = self.params.get('epoch_train_batches', 10)
            validation_steps = self.params.get('epoch_val_batches', None)

        # ignore class weights for regression
        if self.params['loss_fn'] in ['mean_squared_error', 'mean_absolute_error']:
            class_weight = None
        else:
            class_weight = {0: 1, 1: self.params.get('pos_weight', 1.0)}

        with self.sess.as_default():

            self.model.fit(X_train, y_train,
                           batch_size=train_batch_size,
                           epochs=self.params['n_epochs'],
                           verbose=self.params['verbose'],
                           callbacks=callbacks,
                           validation_data=validation_data,
                           shuffle=True,
                           class_weight=class_weight,
                           sample_weight=weights,
                           steps_per_epoch=steps_per_epoch,
                           validation_steps=validation_steps)

            if return_preds:
                chunk_size = int(self.params['predict_chunk_size'])
                fold_preds = self.model.predict(X_val,
                                           batch_size=chunk_size,
                                           verbose=0).ravel()
                return fold_preds


    def _grid_cv_fold(self, fold, plot_n_samples):
        '''Single-fold CV on permutations of cv_grid params'''
        params_grid, keys = self._get_cv_params_grid()
        columns_list = ['fold_no', *keys]
        for met in self.metrics:
            columns_list.extend(['best_' + met, 'rnd_' + met])
        fold_results_list = []
        X_train, y_train, X_val, y_val = self._get_fold_data(fold)

        if self.weights is not None:
            weights = self.get_sample_weights(fold)
        else: weights = None

        for i, param_set in enumerate(params_grid):
            params_str = ''
            for j in range(len(param_set)):
                self.params[keys[j]] = param_set[j]
                params_str += f'{keys[j]}={self.params[keys[j]]}  '
            self.logger.info(params_str)

            self.init_hparams()
            self.init_tensorboard(fold, i+1)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.train_eval(X_train, y_train, X_val, y_val, weights,
                fold=fold, plot_n_samples=plot_n_samples)
            tf.reset_default_graph()

            best_evals = self.best_eval_multi()
            for eval in best_evals:
                self.logger.info(f'   best val {eval[0]}: {eval[1]:.4f}, ' +
                                 f'round {eval[2]}')
            self.logger.info('')
            results_row = [fold, *(str(k) for k in param_set)]
            for eval in best_evals:
                results_row.extend([eval[1], eval[2]])
            round_results = pd.DataFrame([results_row], columns=columns_list, index=[i])
            fold_results_list.append(round_results)

        return pd.concat(fold_results_list, axis=0)


    def grid_cv(self, val_rounds, plot_n_samples=None):
        '''Grid cross-valdidation.  Permutes params/values in self.cv_grid (dict).

        Args: val_rounds, integer: number of CV rounds
            (mimimum: 1, maximum: number of folds)

        Returns: no return; updates self.cv_results with grid CV results
        '''
        self.load_hparams()
        keys = [*self.cv_grid.keys()]
        columns = []
        for met in self.metrics:
            columns.extend(['best_' + met, 'rnd_' + met])
        results_list = []
        self.log_hparams()
        for fold in range(1, val_rounds + 1):
            self.logger.info(f'------------------------ FOLD {fold} OF {val_rounds} ------------------------')
            fold_results = self._grid_cv_fold(fold, plot_n_samples)
            results_list.append(fold_results)

        self.cv_results = pd.concat(results_list, axis=0)
        self.logger.info('grid CV complete.')

        if plot_n_samples is not None:
            self.plot_regression_preds()

        # display/log grid CV summary
        groupby = [self.cv_results[key] for key in keys]
        summ_df = self.cv_results[columns].groupby(groupby).mean()
        self.logger.info(self.parse_summ_df(summ_df))

        # reset/reload all params from params file
        self.load_hparams()


    def cv_predictions(self):
        '''Generate fold-by-fold predictions.  For each fold k, train on all other
        folds and make predictions for k.

        Returns: pandas DataFrame with predictions for each fold in the training set.
        '''
        self.load_hparams()
        self.logger.info(f'starting predictions for CV outputs...')

        train_preds = []

        for fold in range(1, self.n_folds + 1):
            _, val_idx = self._get_fold_indices(fold)
            X_train, y_train, X_val, y_val = self._get_fold_data(fold)

            if self.weights is not None:
                weights = self.get_sample_weights(fold)
            else: weights = None

            self.init_hparams()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            fold_preds = self.train_eval(X_train, y_train, X_val, y_val,
                return_preds=True, fold=fold)
            tf.reset_default_graph()

            fold_preds = pd.Series(fold_preds, index=val_idx)
            train_preds.append(fold_preds)
            self.logger.info(f'fold {fold} CV outputs complete.')
        train_preds = pd.concat(train_preds)
        return train_preds.rename(self.prefix + self.output_suffix, inplace=True)


    def test_predictions(self):
        '''Train on full X_train/y_train and return predictions for X_test
        '''

        if self.weights is not None:
            weights = self.weights
        else: weights = None

        self.load_hparams()
        self.init_hparams()
        self.logger.info(f'starting predictions for test outputs...')

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        test_preds = self.train_eval(self.X_train, self.y_train,
            self.X_test, return_preds=True, weights=weights)
        tf.reset_default_graph()

        test_preds = pd.Series(test_preds, index=self.X_test.index)
        self.logger.info(f'test set outputs complete.')
        return test_preds.rename(self.prefix + self.output_suffix, inplace=True)


class FFNRegressor(FFNModel):

    def __init__(self, X_train, y_train, X_test, params_file=None, folds_lookup=None,
        prefix='keras_regressor', weights=None, tf_path=r'/mltools/tf', logger=None):
        FFNModel.__init__(self, X_train, y_train, X_test, params_file, folds_lookup,
            prefix, weights, tf_path, logger)
        self.metrics = ['MSE', 'MAE']


    def init_hparams(self):
        FFNModel.init_hparams(self)
        self.params['pos_weight'] = 1.0


    def best_eval_multi(self):
        '''Return the minimum value for MSE and MAE
        '''
        return FFNModel.best_eval_multi(self, 'min')


    def build_model(self, enable_metrics=True):

        X, ff_outputs = self.feedforward_layers(final_activation=None)

        model = Model(inputs=X, outputs=ff_outputs)

        metrics = []

        if enable_metrics:
            metrics = ['mean_squared_error', 'mean_absolute_error']

        model.compile(self.optimizer,
                    loss=self.params['loss_fn'],
                    metrics=metrics
        )

        self.model = model


    def train_begin(self, logs):
        self.evals_out = {'round': [], 'lr': [],
                          'train': {'MSE': [], 'MAE': []},
                          'val': {'MSE': [], 'MAE': []}}
        self.logger.info(f'                 TRAIN       |               VAL        |')
        self.logger.info(f'EPOCH       MSE         MAE  |        MSE          MAE  |     eta')


    def epoch_begin(self, epoch, logs):
        return


    def epoch_end(self, epoch, logs, *args):
        '''Callback, log metrics for each epoch'''
        train_mse = logs['mean_squared_error']
        train_mae = logs['mean_absolute_error']
        val_mse = logs['val_mean_squared_error']
        val_mae = logs['val_mean_absolute_error']
        lr = logs['lr']

        self.logger.info(f'{epoch + 1:>4}  {train_mse:9.4f}   {train_mae:9.4f}  |  ' +
                         f'{val_mse:9.4f}    {val_mae:9.4f}  |  {lr:.1e}')

        self.evals_out['round'].append(epoch+1)
        self.evals_out['train']['MSE'].append(train_mse)
        self.evals_out['train']['MAE'].append(train_mae)
        self.evals_out['val']['MSE'].append(val_mse)
        self.evals_out['val']['MAE'].append(val_mae)
        self.evals_out['lr'].append(lr)


    def regression_plot_samples(self, X_val, y_val, n_samples, fold):
        chunk_size = self.params.get('predict_chunk_size', 500)
        sample_idx = np.random.choice(X_val.shape[0], n_samples)
        sample = X_val.iloc[sample_idx, :]
        with self.sess.as_default():
            sample_preds = self.model.predict(sample,
                                              batch_size=chunk_size,
                                              verbose=0).ravel()
        self.sample_preds['pred'][fold] = sample_preds
        self.sample_preds['actual'][fold] = y_val.iloc[sample_idx]


    def train_end(self, X_val, y_val, plot_n_samples, fold):
        self.regression_plot_samples(X_val, y_val, plot_n_samples, fold)



class FFNBinaryClassifier(FFNModel):
    def __init__(self, X_train, y_train, X_test, params_file, folds_lookup=None,
        prefix='keras_classifier', weights=None, tf_path=r'/mltools/tf', logger=None):
        FFNModel.__init__(self, X_train, y_train, X_test, params_file, folds_lookup,
            prefix, weights, tf_path, logger)
        self.y_train = self.y_train.astype(int)
        self.metrics = ['AUC', 'acc', 'precision', 'recall']


    def best_eval_multi(self):
        '''Return the maximum round result for all metrics
                 (AUC, accuracy, precision, and recall)
        '''
        return FFNModel.best_eval_multi(self, 'max')


    def roc_auc(self, y_true, y_pred):
        '''Custom ROC AUC metric from tf.metrics'''
        auc_op, auc_val = tf.metrics.auc(y_true, y_pred)
        auc_vars = [i for i in tf.local_variables() if i.name.startswith('metrics/roc_auc')]
        for v in auc_vars:
            tf.add_to_collection('auc_vars', v)
        with tf.control_dependencies([auc_op]):
            auc_val = tf.identity(auc_val)
            return auc_val


    def build_model(self, enable_metrics=True):
        '''Add binary classification metrics and compile model'''
        X, ff_outputs = self.feedforward_layers(
                                final_activation=tf.keras.activations.sigmoid)

        self.model = Model(inputs=X, outputs=ff_outputs)

        metrics = []

        if enable_metrics and (self.params.get('binary_metrics', 'sklearn') in ['tf', 'both']):
            metrics += [tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.Precision(),
                        tf.keras.metrics.Recall(),
                        self.roc_auc]

        self.model.compile(self.optimizer,
                    loss=self.params['loss_fn'],
                    metrics=metrics
        )


    def train_begin(self, logs):
        '''Initialize self.evals_out and create metrics header for logging'''
        self.evals_out = {'round': [], 'lr': [],
                          'train': {'AUC': [], 'acc': [], 'precision': [], 'recall': []},
                          'val': {'AUC': [], 'acc': [], 'precision': [], 'recall': []}}

        self.logger.info(f'EPOCH              TRAIN             |               VAL')
        self.logger.info(f'         acc     auc    prec  recall |    acc     auc    prec  recall')
        # initialize true_positives, true_negatives, false_positives, false_negatives for AUC
        self.sess.run(tf.variables_initializer(tf.get_collection('auc_vars')))


    def sklearn_binary_metrics(self, X, y):
        '''Accuracy, precision, recall, and AUC from scikit-learn.metrics'''
        batch_size = self.params.get('predict_chunk_size', 1000)
        y_pred_soft = self.model.predict(X, batch_size=batch_size).ravel()
        y_pred_hard = y_pred_soft.round()
        conf = confusion_matrix(y, y_pred_hard)
        acc = (conf[0, 0] + conf[1, 1]) / conf.sum()
        precision = conf[1, 1] / (conf[1, 1] + conf[0, 1])
        recall = conf[1, 1] / (conf[1, 1] + conf[1, 0])
        fpr, tpr, _ = roc_curve(y, y_pred_soft)
        auc_ = auc(fpr, tpr)
        return acc, precision, recall, auc_


    def epoch_begin(self, epoch, logs):
        '''Callbacks: reset true_positives, true_negatives, false_positives,
        false_negatives for AUC
        '''
        self.sess.run(tf.variables_initializer(tf.get_collection('auc_vars')))


    def epoch_end(self, epoch, logs, X_train, y_train, X_val, y_val):
        '''Callbacks: log metrics for each epoch'''
        train_acc, train_prec, train_rec, train_auc =\
            self.sklearn_binary_metrics(X_train, y_train)
        val_acc, val_prec, val_rec, val_auc =\
            self.sklearn_binary_metrics(X_val, y_val)


        metric_choice = self.params.get('binary_metrics', 'sklearn')

        if metric_choice in ['sklearn', 'both']:

            self.logger.info(f'{str(epoch + 1):>4}  {train_acc:.4f}  {train_auc:.4f}  ' +
                             f'{train_prec:.4f}  {train_rec:.4f} | ' +
                             f'{val_acc:.4f}  {val_auc:.4f}  ' +
                             f'{val_prec:.4f}  {val_rec:.4f} skl')

        if metric_choice in ['keras', 'both']:
            train_acc_keras = logs['binary_accuracy']
            train_auc_keras = logs['roc_auc']
            train_prec_keras = logs['precision']
            train_rec_keras = logs['recall']
            val_acc_keras = logs['val_binary_accuracy']
            val_auc_keras = logs['val_roc_auc']
            val_prec_keras = logs['val_precision']
            val_rec_keras = logs['val_recall']

            self.logger.info(f'{str(epoch + 1):>4}  {train_acc_keras:.4f}  {train_auc_keras:.4f}  ' +
                             f'{train_prec_keras:.4f}  {train_rec_keras:.4f} | ' +
                             f'{val_acc_keras:.4f}  {val_auc_keras:.4f}  ' +
                             f'{val_prec_keras:.4f}  {val_rec_keras:.4f} tf')

        self.evals_out['round'].append(epoch+1)
        lr = logs.get('lr', 0)
        self.evals_out['lr'].append(lr)

        if metric_choice in ['sklearn', 'both']:
            self.evals_out['train']['acc'].append(train_acc)
            self.evals_out['train']['AUC'].append(train_auc)
            self.evals_out['train']['precision'].append(train_prec)
            self.evals_out['train']['recall'].append(train_rec)
            self.evals_out['val']['acc'].append(val_acc)
            self.evals_out['val']['AUC'].append(val_auc)
            self.evals_out['val']['precision'].append(val_prec)
            self.evals_out['val']['recall'].append(val_rec)

        if metric_choice == 'keras':
            self.evals_out['train']['acc'].append(train_acc_keras)
            self.evals_out['train']['AUC'].append(train_auc_keras)
            self.evals_out['train']['precision'].append(train_prec_keras)
            self.evals_out['train']['recall'].append(train_rec_keras)
            self.evals_out['val']['acc'].append(val_acc_keras)
            self.evals_out['val']['AUC'].append(val_auc_keras)
            self.evals_out['val']['precision'].append(val_prec_keras)
            self.evals_out['val']['recall'].append(val_rec_keras)


    def train_end(self, epoch, logs, *args):
        return
