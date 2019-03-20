import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from .. import utils
from ..config import cfg
from .base_model import BaseModel


class GBMModel(BaseModel):
    '''MART/gradient boosted machine, using either XGBoost or LightGBM.
    Uses fixed CV folds to allow use in model stacking.
    '''

    def __init__(self, X_train, y_train, X_test, params_file, folds_lookup=None,
                 prefix=None, weights=None, library='xgb', logger=None):
        BaseModel.__init__(self, X_train, y_train, X_test, params_file, folds_lookup,
            prefix, logger)
        self.library = library
        self.weights = weights
        self.clf = None
        self.metrics = None
        self.output_suffix = ['_' + self.library + s for s in ('_pred', '_score')]


    def load_hparams(self):
        '''Load hyperparameters from file'''
        BaseModel.load_hparams(self)
        if self.library == 'xgb':
            self.metrics = self.params['hparams']['eval_metric']
        elif self.library == 'lgbm':
            self.metrics = self.params['hparams']['metric']


    def log_hparams(self):
        base_params = ''
        base_params += f'          -  train_rounds: {self.params.get("train_rounds", "not set")}\n'
        base_params += f'          -  early_stop_rounds: {self.params.get("early_stop_rounds", "not set")}\n'
        for k, v in self.params['hparams'].items():
            base_params += f'          -  {k}: {v}\n'
        self.logger.info(f'base params:\n{base_params}')


    def get_dataset(self, X_train, y_train):
        '''Create library-specific dataset object

        Args:
            X_train: pandas DataFrame, training features
            y_train: pandas Series, training labels/targets

        Returns: xgboost.DMatrix or LightGBM.Dataset, depending on library selected
        '''
        feature_names = [str(i) for i in X_train.columns.values]
        if self.weights is not None:
            weights = self.weights[X_train.index.values]
        else: weights = None
        if self.library == 'xgb':
            dset = xgb.DMatrix(X_train,
                               label=y_train,
                               weight=weights,
                               feature_names=feature_names,
                               missing=np.nan)

        elif self.library == 'lgbm':
            dset = lgb.Dataset(X_train,
                               label=y_train,
                               weight=weights,
                               feature_name='auto',
                               categorical_feature=self.params['cat_feats'],
                               free_raw_data=False)

        else: raise ValueError(f'library \'{self.library}\' not recognized')

        return dset


    def train(self, dset_train):
        '''Train model (self.clf) without evals.'''
        params = self.params['hparams']
        train_rounds = self.params['train_rounds']
        early_stop_rounds = self.params['early_stop_rounds']

        if self.library == 'xgb':
            self.clf = xgb.train(params, dset_train,
                                 num_boost_round=train_rounds)

        elif self.library == 'lgbm':
            self.clf = lgb.train(params, dset_train, train_rounds)

        else: raise ValueError(f'library \'{self.library}\' not recognized')


    def train_eval(self, params, dset_train, dset_val):
        '''Train model (self.clf) and assign round-by-round evals to self.evals_out'''
        train_rounds = self.params['train_rounds']
        early_stop_rounds = self.params['early_stop_rounds']

        if self.library == 'xgb':
            self.clf = xgb.train(params, dset_train,
                           evals=[(dset_train, 'train'), (dset_val, 'val')],
                           evals_result=self.evals_out,
                           num_boost_round=train_rounds,
                           early_stopping_rounds=early_stop_rounds,
                           verbose_eval=self.params['verbose_eval'])

        elif self.library == 'lgbm':
            self.clf = lgb.train(params, dset_train, train_rounds,
                           valid_sets=[dset_train, dset_val],
                           valid_names=['train', 'val'],
                           evals_result=self.evals_out,
                           early_stopping_rounds=early_stop_rounds,
                           verbose_eval=self.params['verbose_eval'])

    def best_eval_multi(self):
        '''Return the best validation and round for each metric in self.metrics
        '''

        #common XGBoost metrics
        min_evals = ['error', 'merror',
                     'rmse', 'mae', 'logloss', 'mlogloss']
        max_evals = ['auc', 'acc', 'map']

        #common LightGBM metrics
        min_evals += ['binary_error', 'multi_error', 'mape',
                      'l2', 'l2_root', 'l1', 'xentropy']

        best_out = []
        for metric in self.metrics:
            if metric in min_evals:
                type='min'
            elif metric in max_evals:
                type='max'
            else: raise ValueError(f'unsupported metric: \'{metric}\', ' +
                f'need min/max specification in best_eval_multi()')
            best_, round_ =  self.best_eval(metric, type)
            best_out.append((metric, best_, round_))
        return best_out


    def predict(self, X_test):
        '''Return predictions from the current model'''

        if self.library == 'xgb':
            dset_test = self.get_dataset(X_test, None)
            preds_p = self.clf.predict(dset_test, output_margin=False)
            preds_margin = self.clf.predict(dset_test, output_margin=True)

        elif self.library == 'lgbm':
            preds_p = self.clf.predict(X_test, raw_score=False)
            preds_margin = self.clf.predict(X_test, raw_score=True)
        return preds_p, preds_margin


    def get_feature_importance(self, sort='weight'):
        '''Feature importance of the current model.

        Args: sort, string in {'weight', 'gain'}, column to sort by.

        Returns: pandas Dataframe, with columns:
            index: feature name
            weight: number of splits on the feature
            gain: improvement in loss from splits on the feature
        '''

        if self.library == 'xgb':
            feat_weight = self.clf.get_score(importance_type='weight')
            feat_gain = self.clf.get_score(importance_type='gain')
            feat_weight_df = pd.DataFrame.from_dict(feat_weight, orient='index',
                columns=['weight'])
            feat_gain_df = pd.DataFrame.from_dict(feat_gain, orient='index',
                columns=['gain'])
            feat_imp_df = pd.merge(feat_weight_df, feat_gain_df, how='outer',
                left_index=True, right_index=True)

        elif self.library == 'lgbm':
            columns = ['feat', 'weight', 'gain']
            feat_weight = self.clf.feature_importance(importance_type='split')
            feat_gain = self.clf.feature_importance(importance_type='gain')
            feat_imp_df = pd.DataFrame(np.c_[self.X_train.columns.values,
                feat_weight, feat_gain], columns=columns)

        return feat_imp_df.sort_values(by=sort, ascending=False)


    def eval(self, fold=1, plot_flag=True):
        '''Single-round train and validation, using fold #1 as the validation set
        '''
        self.load_hparams()
        self.logger.info(f'all params restored from {self.params_file}.')
        train_rounds = self.params['train_rounds']
        early_stop_rounds = self.params['early_stop_rounds']
        X_train, y_train, X_val, y_val = self._get_fold_data(fold)
        dset_train = self.get_dataset(X_train, y_train)
        dset_val = self.get_dataset(X_val, y_val)
        self.logger.info(f'training using \'{self.library}\'.  ' +
                         f'params:\n{self.params}')
        self.train_eval(self.params['hparams'], dset_train, dset_val)

        best_evals = self.best_eval_multi()
        for eval in best_evals:
            self.logger.info(
                f'{self.library}: best validation {eval[0]}: {eval[1]:.5f}, ' +
                f'round {eval[2]}')
        if plot_flag: self.plot_evals()


    def _grid_cv_fold(self, dset_train, dset_val, fold=None):
        '''Train and evaluate a model for a single fold, based on permutations
        of the values in self.cv_grid.  Send the best metrics and rounds to the logger.

        Args:
            dset_train: dataset for the merged train folds
            dset_val: dataset for the validation fold
            fold: fold number, int in {1, ..., .n_folds}

        Returns: pandas DataFrame with best validation metrics and round
        '''
        params_grid, keys = self._get_cv_params_grid()

        columns_list = ['fold_no', *keys]
        for met in self.metrics:
            columns_list.extend(['best_' + met, 'rnd_' + met])

        fold_results_list = []
        for i, param_set in enumerate(params_grid):
            params_str = ''
            for j in range(len(param_set)):
                self.params['hparams'][keys[j]] = param_set[j]
                if isinstance(self.params["hparams"][keys[j]], float):
                    val_str = f'{self.params["hparams"][keys[j]]:.6f}'
                else:
                    val_str = f'{self.params["hparams"][keys[j]]}'
                params_str += f'{keys[j]}=' + val_str + '  '
            self.logger.info(params_str)
            self.train_eval(self.params['hparams'],
                            dset_train,
                            dset_val)
            best_evals = self.best_eval_multi()
            for eval in best_evals:
                self.logger.info(f'   best validation {eval[0]}: {eval[1]:.5f}, ' +
                                 f'round {eval[2]}')
            results_row = [fold, *(str(k) for k in param_set)]
            for eval in best_evals:
                results_row.extend([eval[1], eval[2]])
            round_results = pd.DataFrame([results_row], columns=columns_list, index=[i])
            fold_results_list.append(round_results)

        return pd.concat(fold_results_list, axis=0)


    def regression_plot_samples(self, X_val, y_val, fold, n_samples):
        '''Generate and store fold-by-fold predictions and actual targets to
        self.sample_preds

        Args:
            X_val, y_val: validation features and targets
            n_samples: number of predictions to generate from the validation data
        '''
        sample_idx = np.random.choice(X_val.shape[0], n_samples)
        sample = X_val.iloc[sample_idx, :]
        sample_preds, _ = self.predict(sample)
        self.sample_preds['pred'][fold] = sample_preds
        self.sample_preds['actual'][fold] = y_val.iloc[sample_idx]


    def grid_cv(self, val_rounds=1, plot_n_samples=None):
        '''Train and evaluate models on using fold-by-fold training and validation data
        based on permutations of the values in self.cv_grid.

        Send a summary of best CV results and best round (averaged across folds)
        to the logger.
        '''
        self.load_hparams()
        self.logger.info(f'all params restored from {self.params_file}.')
        keys = [*self.cv_grid.keys()]
        columns = []
        for met in self.metrics:
            columns.extend(['best_' + met, 'rnd_' + met])

        self.logger.info(f'starting grid CV using library \'{self.library}\'.')
        self.log_hparams()

        results_list = []
        for fold in range(1, val_rounds + 1):
            X_train, y_train, X_val, y_val = self._get_fold_data(fold)
            dset_train = self.get_dataset(X_train, y_train)
            dset_val = self.get_dataset(X_val, y_val)
            self.logger.info(f'------------------------ FOLD {fold} OF {val_rounds} ------------------------')
            fold_results = self._grid_cv_fold(dset_train, dset_val, fold=fold)
            results_list.append(fold_results)

            if plot_n_samples is not None:
                self.regression_plot_samples(X_val, y_val, fold, plot_n_samples)

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
        '''Train and generate fold-by-fold cross-validation predictions
        for model validation and stacking.

        Returns: pandas DataFrame with predictions for each training instance.
        Output index matches the index of X_train loaded into the instance.
        '''
        self.logger.info(f'starting predictions for CV outputs...')
        self.load_hparams()
        self.logger.info(f'all params restored from {self.params_file}.')

        output_cols = [self.prefix + c for c in self.output_suffix]

        self.logger.info('generating CV predictions...')
        train_preds = []

        # get predictions for each fold in the training set
        for fold in range(1, self.n_folds + 1):
            X_train, y_train, X_val, _ = self._get_fold_data(fold)
            _, val_idx = self._get_fold_indices(fold)
            dset_train = self.get_dataset(X_train, y_train)
            self.train(dset_train)
            preds_p, preds_margin = self.predict(X_val)
            fold_preds = pd.DataFrame(np.c_[preds_p, preds_margin],
                                      index=val_idx,
                                      columns=output_cols)
            train_preds.append(fold_preds)
            self.logger.info(f'fold {fold} CV outputs complete.')
        self.logger.info('CV outputs complete')
        return pd.concat(train_preds, axis=0)


    def test_predictions(self):
        '''Train on the full X_train, and generate predictions from X_test.

        Returns: pandas DataFrame with predictions for each training instance.
        Output index matches the index of X_test loaded into the instance.
        '''
        self.load_hparams()
        self.logger.info(f'all params restored from {self.params_file}.')

        output_cols = [self.prefix + c for c in self.output_suffix]
        dset_train = self.get_dataset(self.X_train, self.y_train)
        self.logger.info(f'training for {self.params["train_rounds"]} rounds...')
        self.train(dset_train)
        self.logger.info('generating predictions...')
        preds_p, preds_margin = self.predict(self.X_test)
        self.logger.info(f'test set outputs complete.')
        return pd.DataFrame(np.c_[preds_p, preds_margin],
                                  index=self.X_test.index,
                                  columns=output_cols)
