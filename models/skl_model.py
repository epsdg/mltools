import gc
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.decomposition import IncrementalPCA
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.gaussian_process
from sklearn import metrics
from ..utils import map_fn
from ..config import cfg
from .base_model import BaseModel


def init_clf(clf_name, params):
    model = params['models'][clf_name]
    clf = eval(f'{model}()')
    hparams = params[clf_name]['hparams']
    if hparams is not None:
        clf.set_params(**hparams)
    return clf


def cv_worker_fn(args_in):
    '''Worker function for CV train & eval:
        1) fit a classifier to X_train/y_train
        2) predict for X_val
        3) return binary classification metrics (eval mode only)
    '''
    clf_name = args_in['clf_name']
    X_train = args_in['X_train']
    y_train = args_in['y_train']
    X_val = args_in['X_val']
    y_val = args_in['y_val']
    model = args_in['params']['models'][clf_name]
    clf = init_clf(clf_name, args_in['params'])
    if args_in['mode'] == 'eval' and args_in['cv_params'] is not None:
        clf.set_params(**args_in['cv_params'])
    clf.fit(X_train, y_train)
    predict = clf.predict(X_val)
    if args_in['predict_proba']:
        predict_proba = clf.predict_proba(X_val)

    if args_in['mode'] == 'eval':
        columns_list = ['fold_no', *args_in['cv_params'].keys(), 'auc', 'acc']
        acc = metrics.accuracy_score(y_val, predict)
        if args_in['predict_proba']:
            auc  = metrics.roc_auc_score(y_val, predict_proba[:, 1])
        else: auc = np.nan
        results_row = [args_in['fold'],
            *[str(i) for i in args_in['cv_params'].values()], auc, acc]
        round_results = pd.DataFrame([results_row], columns=columns_list)
        gc.collect()
        return round_results

    elif args_in['mode'] == 'outputs':
        outputs_df = pd.DataFrame(predict, columns=[args_in['clf_name'] + '_predict'],
                                  index=X_val.index)
        if args_in['predict_proba']:
            pred_ser = pd.Series(predict_proba[:, 1], index=X_val.index)
            pred_ser = pred_ser.rename(args_in['clf_name'] + '_predict_proba')
            outputs_df = pd.concat([outputs_df, pred_ser], axis=1)
        if args_in['decision_func']:
            dec_func = clf.decision_function(X_val)
            dec_ser = pd.Series(dec_func, index=X_val.index)
            dec_ser = dec_ser.rename(args_in['clf_name'] + '_dec_func')
            outputs_df = pd.concat([outputs_df, dec_ser], axis=1)
        return {'name': args_in['clf_name'], 'output': outputs_df}


class SKLModelSet(BaseModel):
    '''Multi-model classifier using scikit-learn.  Uses fixed CV folds to allow
    use in model stacking.

    Methods:

    1) grid_cv(): Grid cross-validation for a single model.
    2) cv_outputs(): Generate predictions from all models in the model set for
       feeding to the next layer of stacking.
    '''
    def __init__(self, X_train, y_train, X_test, params_file, folds_lookup=None,
                prefix=None, weights=None, backend=None, n_processes=None, logger=None):
        BaseModel.__init__(self, X_train, y_train, X_test, params_file, folds_lookup,
            prefix, logger)

        self.X_train_PCA = None
        self.X_test_PCA = None

        self.default_n_processes = cfg.get('default_n_processes', 2)
        self.n_processes = n_processes
        self.backend = backend


    def preprocess(self, fillna=True, fill_with='mean', standardize=True,
                   clip_outliers=None, use_PCA=True, n_PCA_components=-1):
        '''Preprocessing routine from BaseModel (fill NaNm standardize), and
        generate PCA dataset.
        '''
        BaseModel.preprocess(self, fillna, fill_with, standardize, clip_outliers)

        train_idx = self.X_train.index
        test_idx = self.X_test.index
        cols_in = self.X_train.columns
        train_len = self.X_train.shape[0]

        X = np.concatenate([self.X_train.values, self.X_test.values], axis=0)

        if use_PCA:
            if n_PCA_components == -1: n_PCA_components = X.shape[1]
            self.logger.info('applying PCA...')
            ipca = IncrementalPCA(n_components=n_PCA_components, batch_size=5000)
            X = ipca.fit_transform(X)

        if clip_outliers is not None:
            X = np.where(X>clip_outliers, clip_outliers, X)
            X = np.where(X<-clip_outliers, -clip_outliers, X)

        cols = [f'PCA_{i+1:04d}' for i in range(n_PCA_components)]
        self.X_train_PCA = pd.DataFrame(X[:train_len, :], index=train_idx, columns=cols)
        self.X_test_PCA = pd.DataFrame(X[train_len:, :], index=test_idx, columns=cols)

        self.logger.info('PCA inputs generated.')

        # convert class labels to int
        self.y_train = self.y_train.astype(int)


    def _get_fold_data(self, fold_no, use_pca=False):
        '''Return standard or PCA dataset depending on use_pca setting in params
        '''
        train_idx, test_idx = self._get_fold_indices(fold_no)
        if use_pca:
            X_train = self.X_train_PCA.reindex(train_idx)
            X_val = self.X_train_PCA.reindex(test_idx)
        else:
            X_train = self.X_train.reindex(train_idx)
            X_val = self.X_train.reindex(test_idx)
        y_train = self.y_train.reindex(train_idx)
        y_val = self.y_train.reindex(test_idx)
        return X_train, y_train, X_val, y_val


    def _grid_cv_fold_work_gen(self, clf_name, fold):
        '''Create iterator with work for map_fn
        '''
        use_pca = self.params[clf_name]['use_pca']
        X_train, y_train, X_val, y_val = self._get_fold_data(fold, use_pca)
        predict_proba = self.params[clf_name]['predict_proba']
        decision_func = self.params[clf_name]['decision_func']
        params_grid, keys = self._get_cv_params_grid()

        for param_set in params_grid:
            worker_params = {}
            for j in range(len(self.cv_grid)):
                worker_params[keys[j]] = param_set[j]
            work = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val,
                    'y_val': y_val, 'clf_name': clf_name, 'mode': 'eval',
                    'params': self.params, 'cv_params': worker_params, 'fold': fold,
                    'predict_proba': predict_proba, 'decision_func': decision_func,
                    'use_pca': use_pca}

            yield work


    def log_hparams(self, clf_name):
        base_params = ''
        base_params += f'          -  use_pca: {self.params[clf_name].get("use_pca", "use_pca not set")}\n'
        base_params += f'          -  decision_func: {self.params[clf_name].get("decision_func", "decision_func not set")}\n'
        base_params += f'          -  predict_proba: {self.params[clf_name].get("predict_proba", "predict_proba not set")}\n'
        for k, v in self.params[clf_name]['hparams'].items():
            base_params += f'          -  {k}: {v}\n'
        self.logger.info(f'base params:\n{base_params}')


    def cv_results_summ(self):
        keys = [*self.cv_grid.keys()]
        self.cv_results.auc.fillna(0, inplace=True)
        return self.cv_results.groupby(keys)['acc', 'auc'].mean()


    def grid_cv(self, clf_name, cv_rounds=1, backend=None, n_processes=None):
        '''Grid cross validation'''
        if backend is None:
            if self.backend is None:
                backend = self.params['default_backend']
            else: backend = self.backend

        if n_processes is None:
            if self.n_processes is None:
                n_processes = self.params['default_n_processes']
            else: n_processes = self.n_processes

        self.log_hparams(clf_name)
        work = []
        for i in range(1, cv_rounds+1):
            work.extend(self._grid_cv_fold_work_gen(clf_name, i))
        self.logger.info(f'starting grid CV on \'{clf_name}\' ' +
                         f'using backend \'{backend}\', n_processes={n_processes}')
        results = map_fn(cv_worker_fn, work,
                   n_processes=n_processes, backend=backend)
        self.cv_results = pd.concat(results, axis=0)

        self.logger.info(f'CV results summary:\n{self.cv_results_summ()}')
        self.logger.info(f'CV fold results:\n{self.cv_results.to_string(index=False)}')


    def grid_cv_loop(self, clf_name, cv_rounds=1):
        '''grid_cv without parallelism
        '''
        self.log_hparams(clf_name)
        results = []
        self.logger.info(f'starting grid CV on \'{clf_name}\'...')
        for fold in range(1, cv_rounds+1):
            params_list = self._grid_cv_fold_work_gen(clf_name, fold)
            for i, param_set in enumerate(params_list):
                round_results = cv_worker_fn(param_set)
                self.logger.info(f'fold {fold} of {cv_rounds} param set {i+1} done.')
                results.append(round_results)
        self.cv_results = pd.concat(results, axis=0)

        self.logger.info(f'CV results summary:\n{self.cv_results_summ()}')
        self.logger.info(f'CV fold results:\n{self.cv_results.to_string(index=False)}')


    def _cv_predictions_work_gen(self, use_models):
        for clf_name in use_models:
            use_pca = self.params[clf_name]['use_pca']
            predict_proba = self.params[clf_name]['predict_proba']
            decision_func = self.params[clf_name]['decision_func']
            use_pca = self.params[clf_name]['use_pca']

            # train (folds) data
            for fold in range(self.n_folds):
                X_train, y_train, X_val, y_val = self._get_fold_data(fold, use_pca)
                work = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val,
                    'y_val': y_val, 'clf_name': clf_name, 'mode': 'outputs',
                    'params': self.params, 'cv_params': None, 'fold': fold,
                    'predict_proba': predict_proba, 'decision_func': decision_func,
                    'use_pca': use_pca}
                yield work


    def _test_predictions_work_gen(self, use_models):

        for clf_name in use_models:
            use_pca = self.params[clf_name]['use_pca']
            predict_proba = self.params[clf_name]['predict_proba']
            decision_func = self.params[clf_name]['decision_func']
            use_pca = self.params[clf_name]['use_pca']

            if use_pca:
                X_train = self.X_train_PCA
                X_val = self.X_test_PCA
            elif not use_pca:
                X_train = self.X_train
                X_val = self.X_test
            y_train = self.y_train

            # test data
            work = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val,
                'y_val': None, 'clf_name': clf_name, 'mode': 'outputs',
                'params': self.params, 'cv_params': None, 'fold': 'test',
                'predict_proba': predict_proba, 'decision_func': decision_func,
                'use_pca': use_pca}
            yield work


    def _join_predictions(self, list_in, use_models):
        '''Concatenate predictions outputs into a single DataFrame
        '''
        outputs_dict = {k: [] for k in use_models}
        for o in list_in: outputs_dict[o['name']].append(o['output'])
        outputs_list = []
        for df_list in outputs_dict.values():
            df = pd.concat(df_list, axis=0)
            outputs_list.append(df)
        return pd.concat(outputs_list, axis=1)


    def _predictions(self, mode, backend=None, n_processes=None):
        '''Parallel routine for train/test predictions'''
        if backend is None:
            if self.backend is None:
                backend = self.params['default_backend']
            else: backend = self.backend

        if n_processes is None:
            if self.n_processes is None:
                n_processes = self.params['default_n_processes']
            else: n_processes = self.n_processes

        self.load_hparams()
        self.logger.info(f'all params restored from ./{self.params_file}')

        self.logger.info(f'backend=\'{backend}\', n_processes={n_processes}')
        use_models = self.params['use_models']

        if mode=='cv':
            work_gen_fn = self._cv_predictions_work_gen
        if mode=='test':
            work_gen_fn = self._test_predictions_work_gen

        results_list = map_fn(cv_worker_fn,
                       work_gen_fn(use_models),
                       n_processes=n_processes, backend=backend)

        return self._join_predictions(results_list, use_models)


    def cv_predictions(self, backend=None, n_processes=None):
        '''Generate fold-by-fold predictions.  For each fold k, train on all other
        folds and make predictions for k.

        Returns: pandas DataFrame with predictions for each fold in the training set.
        '''
        self.logger.info('Stating CV predictions...')
        return self._predictions('cv', backend, n_processes)


    def test_predictions(self, backend=None, n_processes=None):
        '''Train on full X_train/y_train and return predictions for X_test
        '''
        self.logger.info('Stating test set predictions...')
        return self._predictions('test', backend, n_processes)


    def _predictions_loop(self, mode):
        '''Predictions routine for cv_predictions_loop and test_predictions_loop
        '''
        self.load_hparams()
        self.logger.info(f'all params restored from ./{self.params_file}')
        results_list = []
        use_models = self.params['use_models']

        if mode=='cv':
            work_gen_fn = self._cv_predictions_work_gen
        if mode=='test':
            work_gen_fn = self._test_predictions_work_gen

        work = work_gen_fn(use_models)

        for obj in work:
            results = cv_worker_fn(obj)
            results_list.append(results)
        return self._join_predictions(results_list, use_models)


    def cv_predictions_loop(self):
        '''Generate fold-by-fold predictions.  For each fold k, train on all other
        folds and make predictions for k (non-parallel version).

        Returns: pandas DataFrame with predictions for each fold in the training set.
        '''

        self.logger.info('Stating CV predictions...')
        return self._predictions_loop('cv')


    def test_predictions_loop(self):
        '''Train on full X_train/y_train and return predictions for X_test
        (non-parallel version)'''
        self.logger.info('Stating test set predictions...')
        return self._predictions_loop('test')
