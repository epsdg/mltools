import numpy as np
import pandas as pd
from .. import utils
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer


class BaseModel():
    def __init__(self, X_train, y_train, X_test, params_file, folds_lookup,
                 prefix, logger):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.params_file = params_file
        self.params = None
        self.evals_out = {}  # for GBMModel and DNN only
        self.cv_grid = {}
        self.cv_results = None
        self.sample_preds = {'pred': {}, 'actual': {}}
        self.prefix = prefix

        # generate default logger if none is passed
        if logger is None:
            self.logger = utils.get_logger('mltools')
        elif isinstance(logger, str):
             self.logger = utils.get_logger(logger)
        else:
            self.logger = logger

        if isinstance(folds_lookup, int):
            self.logger.info(f'generating {folds_lookup} random folds...')
            self.folds_lookup = self.generate_folds(folds_lookup)
        if folds_lookup is None:
            self.logger.info(f'generating 5 random folds...')
            self.folds_lookup = self.generate_folds(5)
        else: self.folds_lookup = folds_lookup
        self.folds_lookup = self.folds_lookup.astype(int)
        self.n_folds = len(self.folds_lookup.unique())


    def load_hparams(self):
        '''Load hyperparameters and other settings from yaml file'''
        if self.params_file is not None:
            self.params = utils.load_hparams(self.params_file)
        else: print('params_file not set.')


    def generate_folds(self, n_folds=5):
        n_samples = self.X_train.shape[0]
        folds = []
        for i in range(n_folds):
            folds.extend([i+1] * (n_samples // n_folds))
        folds.extend([i+1 for i in range(n_samples % n_folds)])
        return pd.Series(np.random.permutation(folds), index=self.X_train.index)


    def log_hparams(self):
        '''Send a list of hyperparameters to the logger'''
        base_params = ''
        for k, v in self.params.items():
            base_params += f'          -  {k}: {v}\n'
        self.logger.info(f'base params:\n{base_params}')


    def preprocess(self, fillna=True, fill_with='mean', standardize=True,
                    clip_outliers=None, *kargs):
        '''Fill NaN, standardize inputs, and apply min/max clip for outliers'''
        self.logger.info('preprocessing inputs:')
        train_idx = self.X_train.index
        test_idx = self.X_test.index
        cols_in = self.X_train.columns
        train_len = self.X_train.shape[0]

        X = np.concatenate([self.X_train.values, self.X_test.values], axis=0)

        if fillna:
            imputer = SimpleImputer(strategy=fill_with, verbose=1)
            self.logger.info('  filling NaN...')
            X[X == np.inf] = np.nan
            X[X == -np.inf] = np.nan
            X = imputer.fit_transform(X)

        if standardize:
            self.logger.info('  standardizing inputs...')
            X = scale(X)

        if clip_outliers is not None:
            X = np.where(X>clip_outliers, clip_outliers, X)
            X = np.where(X<-clip_outliers, -clip_outliers, X)

        self.X_train = pd.DataFrame(X[:train_len, :], index=train_idx, columns=cols_in)
        self.X_test = pd.DataFrame(X[train_len:, :], index=test_idx, columns=cols_in)
        self.logger.info('  finished.')


    def plot_results(self, filename='evals_plot.png'):
        import matplotlib.pyplot as plt
        titles = list(self.evals_out['train'].keys())
        n_plots = len(titles)
        fig = plt.figure(figsize=(n_plots * 4, 3))
        for i, title in enumerate(titles):
            ax = fig.add_subplot(1, n_plots, i + 1)
            plt.plot(self.evals_out['val'][title], color='r')
            plt.plot(self.evals_out['train'][title], color='b')
            plt.title(title);
        plt.savefig(filename)


    def plot_regression_preds(self):
        import matplotlib.pyplot as plt
        colors = ['r', 'm', 'g', 'c', 'b']
        all_act = []
        figure = plt.figure(figsize=(5, 5))
        for i, k in enumerate(self.sample_preds['pred'].keys()):
            all_act.extend(self.sample_preds['actual'][k])
            plt.scatter(self.sample_preds['actual'][k],
                        self.sample_preds['pred'][k],
                        color=colors[i], alpha=0.5, s=4)
        min_, max_ = min(all_act), max(all_act)
        plt.plot((min_, max_), (min_, max_), color='k')
        plt.xlabel('actual value', fontsize=12)
        plt.ylabel('predicted value', fontsize=12);
        plt.savefig('CV_preds_plot.png')


    def best_eval(self, eval_name, type='max'):
        eval_list = self.evals_out['val'][eval_name]
        if type == 'max': best_eval = max(eval_list)
        elif type == 'min': best_eval = min(eval_list)
        if 'round' in self.evals_out:
            best_round = self.evals_out['round'][eval_list.index(best_eval)]
        else: best_round = eval_list.index(best_eval) + 1
        return best_eval, best_round


    def best_eval_multi(self, type):
        best_out = []
        if self.metrics is not None:
            for metric in self.metrics:
                best_, round_ =  self.best_eval(metric, type)
                best_out.append((metric, best_, round_))
        else:
            raise ValueError('self.metrics must not be empty')
        return best_out


    def evals_df(self):
        '''Transform round-by-round metrics from dict to pd.DataFrame'''
        parts = []
        for pref in ['train', 'val']:
            metrics = list(self.evals_out[pref].keys())
            columns = {metric: pref + '_' + tag for metric in metrics}
            df = pd.DataFrame.from_dict(dict_in[pref]).rename(columns=columns)
            parts.append(df)
        return pd.concat(parts, axis=1)


    def parse_summ_df(self, df_in):
        '''Args: df_in, pd.DataFrame with summary CV results
        Returns: fixed-width string with metrics for the logger
        '''
        r = df_in.iterrows()
        if isinstance(next(r)[0], tuple):
            n_groups = len(next(r)[0])
        else:
            n_groups = 1

        outstr = 'CV results summary, validation scores:\n'
        header = ''
        for v in df_in.index.names:
            header += f'{v:<14}'
        for v in df_in.columns.values:
            header += f'{v:>12}'
        outstr += header + '\n'

        rows = df_in.iterrows()
        for row in rows:
            rowstr = ''
            if n_groups == 1:
                rowstr += f'{str(row[0]):<14}'
            elif n_groups > 1:
                for v in row[0]:
                    rowstr += f'{str(v):<14}'
            for v in row[1].values:
                rowstr += f'{v:>12.4f}'
            outstr += rowstr + '\n'

        return outstr


    def _get_fold_indices(self, i):
        fold_idx = []
        for j in range(1, self.n_folds + 1):
            fold_idx = fold_idx + [self.folds_lookup[self.folds_lookup == j].index]
        test_idx = fold_idx.pop(i-1)
        train_idx = []
        for idx in fold_idx:
            train_idx.extend(idx.values)
        return pd.Index(train_idx), test_idx


    def _get_fold_data(self, fold_no):
        train_idx, test_idx = self._get_fold_indices(fold_no)
        X_train = self.X_train.reindex(train_idx)
        X_val = self.X_train.reindex(test_idx)
        y_train = self.y_train.reindex(train_idx)
        y_val = self.y_train.reindex(test_idx)
        return X_train, y_train, X_val, y_val


    def _get_cv_params_grid(self):
        '''Transforms self.cv_grid (dict) into permuted parameter sets
        for the CV routine.

        Returns:
            params_grid: list of permuted param values. Each item is a
                list of param values, ordered by key (param name).
            keys: list, param names
        '''
        keys = list(self.cv_grid.keys())
        params_grid = []
        for v in self.cv_grid.values():
            if params_grid == []:
                for setting in v:
                    params_grid.append([setting])
            else:
                params_grid_ = params_grid.copy()
                params_grid = []
                for setting in v:
                    for j in params_grid_:
                        params_grid.append(j + [setting])
        return params_grid, keys
