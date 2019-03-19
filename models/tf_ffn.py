import gc
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial
import tensorflow as tf
from sklearn import preprocessing
from .. import utils
from ..config import cfg
from .base_model import BaseModel

class Graph():
    '''Container class for tf.Graph and associated variables.
    '''
    def __init__(self):
        self.graph = None


class FFNModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, params_file=None, folds_lookup=None,
        prefix=None, tf_path=None, logger=None):
        BaseModel.__init__(self, X_train, y_train, X_test, params_file, folds_lookup,
            prefix, logger)
        self.graph = None

        self.n_inputs = None
        self.n_outputs = None
        self.initializer = None
        self.regularizer = None
        self.activation = None

        self.tf_path = tf_path
        self.logdir = None
        self.output_suffix = '_dnn_pred'


    def preprocess(self, imputer_strategy='mean'):
        '''Mean-fill NaN, center, and scale inputs
        '''
        train_idx = self.X_train.index
        test_idx = self.X_test.index
        cols_in = self.X_train.columns
        train_len = self.X_train.shape[0]

        X = np.concatenate([self.X_train.values, self.X_test.values], axis=0)

        imputer = preprocessing.Imputer(strategy=imputer_strategy, axis=0, verbose=1)
        self.logger.info('filling NaN...')
        X[X == np.inf] = np.nan
        X[X == -np.inf] = np.nan
        X = imputer.fit_transform(X)

        self.logger.info('standardizing inputs...')
        X = preprocessing.scale(X)

        self.X_train = pd.DataFrame(X[:train_len, :], index=train_idx, columns=cols_in)
        self.X_test = pd.DataFrame(X[train_len:, :], index=test_idx, columns=cols_in)
        del X
        self.logger.info('preprocessing complete.')


    def init_hparams(self):
        '''interpret params.yaml file to set tf.Graph params
        '''
        self.n_inputs = self.X_train.shape[1]

        if 'n_outputs' in self.params:
            self.n_outputs = self.params['n_outputs']
        else:
            self.n_outputs = 1

        if 'init_mode' in self.params:
            init_mode = self.params['init_mode']
        else:
            init_mode = 'FAN_AVG'

        if 'init_uniform' in self.params:
            init_uniform = self.params['init_uniform']
        else:
            init_uniform = True

        self.initializer = (
            tf.contrib.layers
            .variance_scaling_initializer(mode=init_mode,
                                          uniform=init_uniform))

        if 'l1_reg_weight' in self.params:
            l1_reg = float(self.params['l1_reg_weight'])
        else:
            l1_reg = 0.0

        if 'l2_reg_weight' in self.params:
            l2_reg = float(self.params['l2_reg_weight'])
        else:
            l2_reg = 0.0

        reg={'None': None,
             'l1': tf.contrib.layers.l1_regularizer(scale=l1_reg),
             'l2': tf.contrib.layers.l2_regularizer(scale=l2_reg),
             'l1-l2': tf.contrib.layers.l1_l2_regularizer(
                            scale_l1=l1_reg, scale_l2=l2_reg)}

        if 'regularizer' in self.params:
            self.regularizer = reg[self.params['regularizer']]
        else:
            self.regularizer = None

        act={'elu': tf.nn.elu,
             'relu': tf.nn.relu,
             'leaky-relu': tf.nn.leaky_relu,
             'selu': tf.nn.selu,
             'crelu': tf.nn.crelu,
             'tanh': tf.tanh,
             'sigmoid': tf.sigmoid}

        if 'activation' in self.params:
            self.activation = act[self.params['activation']]
        else:
            self.activation = tf.nn.relu
            self.logger.info(f'Activation not specified in params.  ' +
                             f'Using ReLU.')

        optimizers = {
            'sgd': tf.train.GradientDescentOptimizer,
            'momentum': partial(tf.train.MomentumOptimizer,
                                momentum=float(self.params['momentum'])),
            'adam': partial(tf.train.AdamOptimizer,
                             beta1=float(self.params['adam_beta1']),
                             beta2=float(self.params['adam_beta2']),
                             epsilon=float(self.params['adam_epsilon'])),
            'adagrad': tf.train.AdagradOptimizer,
            'adadelta': tf.train.AdadeltaOptimizer,
            'adamw': partial(tf.contrib.opt.AdamWOptimizer,
                             beta1=float(self.params['adam_beta1']),
                             beta2=float(self.params['adam_beta2']),
                             epsilon=float(self.params['adam_epsilon']),
                             weight_decay=float(self.params['adam_weight_decay']))}

        if 'optimizer' in self.params:
            self.optimizer = optimizers[self.params['optimizer']]
        else:
            self.optimizer = tf.train.GradientDescentOptimizer
            self.logger.info(f'Optimizer not specified in params.  ' +
                             f'Using GradientDescentOptimizer')


    def _shuffle_idx(self, X):
        '''Shuffle batch order when training with minibatches.
        '''
        idx = X.index.values
        rng = np.random.RandomState(datetime.now().microsecond)
        return rng.permutation(idx)


    def get_batch(self, X_in, y_in, idx, batch_size, batch_no):
        '''Used in train_mode='minibatch', i.e. each epoch trains against
        full training set (shuffled).
        '''
        idx_batch = idx[batch_size * (batch_no-1):batch_size * batch_no]
        X_batch = X_in.reindex(idx_batch).values
        y_batch = y_in.reindex(idx_batch).values
        return X_batch, y_batch


    def get_sample(self, X_in, y_in, batch_size):
        rng = np.random.RandomState(datetime.now().microsecond)
        idx_in = X_in.index.values
        idx_sample = rng.choice(idx_in, size=batch_size, replace=False)
        X_batch = X_in.loc[idx_sample, :].values
        y_batch = y_in.loc[idx_sample].values
        return X_batch, y_batch


    def init_tensorboard(self):
        '''set directory and filename for tensorboard logs and checkpoint file
        '''
        now = datetime.now().strftime("%m%d-%H%M")
        comment = self.prefix + ''
        self.logdir = f'{self.tf_path}/tensorboard_logs/{now}{comment}/'
        self.ckpt_file = f'{self.tf_path}/sessions/{self.prefix}_tf_model.ckpt'


    def ff_layer(self, g, layer_in, layer_no):

        with g.graph.as_default():

            layer = tf.layers.dropout(layer_in,
                                     rate=self.params['drop_rates'][layer_no],
                                     training=g.train_flag,
                                     name='drop_' + str(layer_no + 1))

            layer = tf.layers.dense(layer,
                                    self.params['layers'][layer_no],
                                    kernel_initializer=self.initializer,
                                    kernel_regularizer=self.regularizer,
                                    name='dense_' + str(layer_no + 1))

            if self.params['use_batch_norm'][layer_no]:
                layer = tf.layers.batch_normalization(
                            layer, training=g.train_flag,
                            momentum=self.params['batch_norm_momentum'],
                            name='bn_' + str(layer_no + 1))

            layer = self.activation(layer, name='act_' + str(layer_no + 1))

            return g, layer


    def _grid_cv_fold(self, fold):
        params_grid, keys = self._get_cv_params_grid()
        columns_list = ['fold_no', *keys]
        for met in self.metrics:
            columns_list.extend(['best_' + met, 'rnd_' + met])
        fold_results_list = []
        X_train, y_train, X_val, y_val = self._get_fold_data(fold)

        for i, param_set in enumerate(params_grid):
            params_str = ''
            for j in range(len(param_set)):
                self.params[keys[j]] = param_set[j]
                params_str += f'{keys[j]}={self.params[keys[j]]}  '
            self.logger.info(params_str)

            self.init_hparams()
            self.train_eval(X_train, y_train, X_val, y_val)
            self.sess.close()
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


    def grid_cv(self, val_rounds):
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
        self.logger.info(f'starting grid CV.')
        self.logger.info(f'base params: {self.params}')
        for fold in range(1, val_rounds + 1):
            self.logger.info(f'------------ FOLD {fold} OF {val_rounds} ------------')
            fold_results = self._grid_cv_fold(fold)
            results_list.append(fold_results)
        self.cv_results = pd.concat(results_list, axis=0)

        # display/log grid CV summary
        groupby = [self.cv_results[key] for key in keys]
        summ_df = self.cv_results[columns].groupby(groupby).mean()
        self.logger.info(self.parse_summ_df(summ_df))

        # reset/reload all params from params file
        self.load_hparams()


    def cv_predictions(self):
        '''Generate fold-by-fold predictions.  For each fold k, train on all other
        folds and make predictions for k.  For test set, train on the full training
        dataset.

        Loads all hyperparameters from the params.yaml file.  Will overwrite any/all
        instance.params settings.

        Args: none.

        Returns: pandas DataFrame with predictions for each fold in the training set,
        combined with predictions for the test set.
        '''
        self.logger.info(f'starting predictions for CV outputs...')
        self.load_hparams()
        self.logger.info(f'all params restored from {self.params_file}.')

        train_preds = []

        for fold in range(1, self.n_folds + 1):
            _, val_idx = self._get_fold_indices(fold)
            X_train, y_train, X_val, y_val = self._get_fold_data(fold)
            fold_outputs = self.train_eval(X_train, y_train, X_val, y_val, return_preds=True)
            self.sess.close()
            preds_ser = pd.Series(fold_outputs, index=val_idx)
            train_preds.append(preds_ser)
            self.logger.info(f'fold {fold} CV outputs complete.')
        train_preds = pd.concat(train_preds)
        return train_preds.rename(self.prefix + self.output_suffix, inplace=True)


    def test_predictions(self):
        test_preds = self.train_eval(self.X_train, self.y_train,
            self.X_test, None, return_preds=True)
        self.sess.close()
        test_preds = pd.Series(test_preds, index=self.X_test.index)
        self.logger.info(f'test set outputs complete.')
        return test_preds.rename(self.prefix + self.output_suffix, inplace=True)


class DNNRegressor(FFNModel):

    def __init__(self, X_train, y_train, X_test, params_file=None, folds_lookup=None,
        prefix=None, weights, tf_path=None, logger=None):
        FFNModel.__init__(self, X_train, y_train, X_test, params_file, folds_lookup,
            prefix, tf_path, logger)
        self.metrics = ['MSE', 'MAE']
        self.n_outputs = 1


    def init_hparams(self):
        FFNModel.init_hparams(self)
        self.n_outputs = 1


    def best_eval_multi(self):
        '''Return the minimum value for MSE and MAE
        '''
        return FFNModel.best_eval_multi(self, 'min')


    def build_graph(self):
        self.init_hparams()
        self.init_tensorboard()
        g = Graph()
        g.graph = tf.Graph()
        with g.graph.as_default():
            g.X = tf.placeholder(tf.float32, shape=(None,
                                                  self.n_inputs),
                                                  name='X')
            g.y = tf.placeholder(tf.float32, shape=(None), name='y')
            g.train_flag = tf.placeholder_with_default(False, shape=(), name='training')

            g.stack = [g.X]

            for layer_no, layer in enumerate(self.params['layers']):
                g, layer_out = self.ff_layer(g, g.stack[-1], layer_no)
                g.stack.append(layer_out)

            g.drop = tf.layers.dropout(g.stack[-1],
                                     rate=self.params['drop_rates'][-1],
                                     training=g.train_flag,
                                     name='drop_before_logits')

            g.dnn_outputs = tf.layers.dense(g.drop, 1, activation=None)

            with tf.name_scope('loss'):
                g.MAE = tf.reduce_mean(tf.abs(g.dnn_outputs - g.y))
                g.MSE = tf.reduce_mean(tf.square(g.dnn_outputs - g.y))
                g.exp_error = tf.reduce_mean(tf.subtract(tf.exp(tf.abs(g.dnn_outputs - g.y)), 1))
                g.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            with tf.name_scope('train'):
                g.optimizer = self.optimizer(learning_rate=float(self.params['eta']))
                objective = self.params['objective']
                if objective == 'MAE':
                    g.loss=tf.add_n([g.MAE] + g.reg_losses, name='combined_loss')
                elif objective == 'MSE':
                    g.loss=tf.add_n([g.MSE] + g.reg_losses, name='combined_loss')
                elif objective == 'exp_error':
                    g.loss=tf.add_n([g.exp_error] + g.reg_losses, name='combined_loss')
                g.training_op = g.optimizer.minimize(g.loss)

            g.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            if self.params['regularizer'] != 'None':
                with tf.name_scope('reg_losses'):
                    g.train_reg_loss = tf.summary.scalar('train', tf.add_n(g.reg_losses))
                    g.val_reg_loss = tf.summary.scalar('val', tf.add_n(g.reg_losses))

            with tf.name_scope('MSE'):
                g.train_mse = tf.summary.scalar('train', g.MSE)
                g.val_mse = tf.summary.scalar('val', g.MSE)

            with tf.name_scope('MAE'):
                g.train_mae = tf.summary.scalar('train', g.MAE)
                g.val_mae = tf.summary.scalar('val', g.MAE)

            g.file_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())

            g.saver = tf.train.Saver()

            return g

    def train_eval(self, X_train, y_train, X_val, y_val, return_preds=False, save_ckpt=False):

        g = self.build_graph()

        self.evals_out = {'round': [],
                          'train': {'MSE': [], 'MAE': []},
                          'val': {'MSE': [], 'MAE': []}}

        train_batch_size = self.params['train_batch_size']
        val_batch_size = self.params['val_batch_size']
        n_val_batches = self.params['n_val_batches']

        if not return_preds:
            # add header for logger
            self.logger.info(f'  RND            TRAIN         |             VAL')
            self.logger.info(f'            MSE         MAE    |      MSE          MAE')

        self.sess = tf.InteractiveSession(graph=g.graph,
                        config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

        if self.params['train_mode'] == 'minibatch':
            n_batches = (X_train.shape[0] // train_batch_size) + 1
            train_batch_size = X_train.shape[0] // n_batches
            self.logger.info(f'CV batch size scaled: {train_batch_size} n_batches {n_batches}')
            self.params['tboard_evals_step'] = 1
            self.params['log_evals_step'] = 1
            self.logger.info(f'evals set to every epoch')
        for epoch in range(self.params['n_epochs']):
            if self.params['train_mode'] == 'minibatch':
                idx = self._shuffle_idx(X_train)
                for batch in range(1, n_batches+1):
                    X_train_batch, y_train_batch = self.get_batch(X_train,
                                        y_train, idx, train_batch_size, batch)
                    train_op_dict = {g.X: X_train_batch, g.y: y_train_batch, g.train_flag:True}
                    self.sess.run([g.training_op, g.extra_update_ops], feed_dict=train_op_dict)

            elif self.params['train_mode'] == 'sgd':
                X_train_batch, y_train_batch = self.get_sample(
                    X_train, y_train, batch_size)
                train_op_dict = {g.X: X_train_batch,
                                 g.y: y_train_batch,
                                 g.train_flag:True}
                self.sess.run([g.training_op, g.extra_update_ops], feed_dict=train_op_dict)

            if ((epoch + 1) % self.params['tboard_evals_step'] == 0
                and not return_preds):

                train_mse_summ = g.train_mse.eval(
                    feed_dict={g.X: X_train_batch, g.y: y_train_batch, g.train_flag:False})
                train_mae_summ = g.train_mae.eval(
                    feed_dict={g.X: X_train_batch, g.y: y_train_batch, g.train_flag:False})
                g.file_writer.add_summary(train_mse_summ, epoch+1)
                g.file_writer.add_summary(train_mae_summ, epoch+1)

                X_val_batch, y_val_batch = self.get_sample(X_val, y_val, val_batch_size)

                val_mse_summ = g.val_mse.eval(
                    feed_dict={g.X: X_val_batch, g.y: y_val_batch, g.train_flag:False})
                val_mae_summ =g. val_mae.eval(
                    feed_dict={g.X: X_val_batch, g.y: y_val_batch, g.train_flag:False})
                g.file_writer.add_summary(val_mse_summ, epoch+1)
                g.file_writer.add_summary(val_mae_summ, epoch+1)

                if self.params['regularizer'] in ['l1', 'l2', 'l1-l2']:
                    train_reg_loss_summ = g.train_reg_loss.eval(
                        feed_dict={g.X: X_train_batch, g.y: y_train_batch, g.train_flag:False})
                    g.file_writer.add_summary(train_reg_loss_summ, epoch)
                    val_reg_loss_summ = g.val_reg_loss.eval(
                        feed_dict={g.X: X_val_batch, g.y: y_val_batch, g.train_flag:False})
                    g.file_writer.add_summary(val_reg_loss_summ, epoch)

            if ((epoch + 1) % self.params['log_evals_step'] == 0
                and not return_preds):

                round_evals = {'train': {'MSE': [], 'MAE': []},
                               'val': {'MSE': [], 'MAE': []}}
                for i in range(n_val_batches):

                    X_train_batch, y_train_batch = self.get_sample(
                                            X_train, y_train, train_batch_size)

                    round_evals['train']['MSE'].append(
                        g.MSE.eval(feed_dict={g.X: X_train_batch,
                                            g.y: y_train_batch,
                                            g.train_flag:False}))

                    round_evals['train']['MAE'].append(
                        g.MAE.eval(feed_dict={g.X: X_train_batch,
                                            g.y: y_train_batch,
                                            g.train_flag:False}))

                    X_val_batch, y_val_batch = self.get_sample(
                                            X_val, y_val, val_batch_size)

                    round_evals['val']['MSE'].append(
                        g.MSE.eval(feed_dict={g.X: X_val_batch,
                                            g.y: y_val_batch,
                                            g.train_flag:False}))

                    round_evals['val']['MAE'].append(
                        g.MAE.eval(feed_dict={g.X: X_val_batch,
                                            g.y: y_val_batch,
                                            g.train_flag:False}))

                train_mse_ = sum(round_evals['train']['MSE']) / n_val_batches
                train_mae_ = sum(round_evals['train']['MAE']) / n_val_batches
                eval_mse_ = sum(round_evals['val']['MSE']) / n_val_batches
                eval_mae_ = sum(round_evals['val']['MAE']) / n_val_batches

                # add round results for logger
                self.logger.info(f' {str(epoch + 1):>4}  {train_mse_:>10.4f}  ' +
                                 f'{train_mae_:>10.4f}  | ' +
                                 f'{eval_mse_:>10.4f}   {eval_mae_:>10.4f}')

                self.evals_out['round'].append(epoch + 1)
                self.evals_out['train']['MSE'].append(train_mse_)
                self.evals_out['train']['MAE'].append(train_mae_)
                self.evals_out['val']['MSE'].append(eval_mse_)
                self.evals_out['val']['MAE'].append(eval_mae_)

        if save_ckpt:
            save_path = g.saver.save(self.sess, self.ckpt_file)
            g.file_writer.close()
            self.logger.info(f'checkpoint saved as \'{self.ckpt_file}\'.')

        if return_preds:
            chunk_size = int(self.params['predict_chunk_size'])
            n_chunks = X_val.shape[0] // chunk_size + 1
            fold_preds = []
            for i in range(n_chunks):
                feed_dict={train_flag:False,
                    X: X_val.iloc[(i*chunk_size):((i+1)*chunk_size), :].values}
                preds_chunk = g.dnn_outputs.eval(feed_dict=feed_dict)
                fold_preds.extend(preds_chunk.ravel())
            return fold_preds


class FFNClassifier(DNNModel):
    def __init__(self, X_train, y_train, X_test, params_file=None, folds_lookup=None,
        prefix=None, weights=None, tf_path=None, logger=None):
        DNNModel.__init__(self, X_train, y_train, X_test, params_file, folds_lookup,
            prefix, tf_path, logger)
        self.y_train = self.y_train.astype(int)
        self.metrics = ['AUC', 'acc', 'precision', 'recall']
        self.ckpt_file = ckpt_file


    def init_hparams(self):
        FFNModel.init_hparams(self)
        if 'pos_weight' not in self.params:
            self.params['pos_weight'] = 1.0


    def best_eval_multi(self):
        '''Return the maximum round result for all metrics
                 (AUC, accuracy, precision, and recall)
        '''
        return FFNModel.best_eval_multi(self, 'max')


    def build_graph(self):
        self.init_hparams()
        self.init_tensorboard()
        g = Graph()
        g.graph = tf.Graph()
        with g.graph.as_default():
            g.X = tf.placeholder(tf.float32, shape=(None, int(self.n_inputs)), name='X')

            if self.n_outputs == 1:
                g.y = tf.placeholder(tf.int32,
                                   shape=(None),
                                   name='y')
                g.y_2d = tf.one_hot(g.y, 2, axis=-1)
            else:
                g.y = tf.placeholder(tf.int32,
                                   shape=(None, int(self.n_outputs)),
                                   name='y')
                g.y_2d = tf.identity(g.y, name='y_passthru')

            g.train_flag = tf.placeholder_with_default(False, shape=(), name='training')

            g.stack = [g.X]

            for layer_no, layer in enumerate(self.params['layers']):
                g, layer_out = self.ff_layer(g, g.stack[-1], layer_no)
                g.stack.append(layer_out)

            g.drop_final = tf.layers.dropout(g.stack[-1],
                                     rate=self.params['drop_rates'][-1],
                                     training=g.train_flag,
                                     name='drop_before_logits')

            if self.n_outputs == 1:
                g.logits =  tf.layers.dense(g.drop_final, 2, name='logits')
            else:
                g.logits =  tf.layers.dense(g.drop_final, int(self.n_outputs), name='logits')

            with tf.name_scope('predictions'):

                g.soft_preds_sparse = tf.nn.softmax(g.logits, name='soft_preds_sparse')

                # TODO: adjust for multi-class
                g.soft_preds_scalar = g.soft_preds_sparse[:, 1]
                #g.soft_preds = tf.slice(g.soft_preds, [0, 1], [-1, 1])
                g.hard_preds_scalar = tf.argmax(g.logits, axis=-1, name='hard_preds_scalar')

                if self.n_outputs == 1:
                    g.hard_preds_sparse = tf.one_hot(g.hard_preds_scalar, 2,
                                                 name='hard_preds_sparse')
                else:
                    g.hard_preds_sparse = tf.one_hot(g.hard_preds_scalar,
                                                 self.n_outputs,
                                                 name='hard_preds_sparse')

            with tf.name_scope('loss'):
                g.xentropy = tf.nn.weighted_cross_entropy_with_logits(g.y_2d,
                    logits=g.logits, pos_weight=self.params['pos_weight'])
                g.xentropy_mean=tf.reduce_mean(g.xentropy, name='xentropy')
                g.reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                g.combined_loss=tf.add_n([g.xentropy_mean] + g.reg_losses, name='combined_loss')

            # BINARY classification: tf.metrics 'accuracy', 'auc', 'precision', and 'recall'
            if self.n_outputs == 1:
                with tf.name_scope('binary_metrics'):
                    g.train_acc_val, g.train_acc_op = tf.metrics.accuracy(
                        labels=g.y, predictions=g.hard_preds_scalar)
                    g.train_auc_val, g.train_auc_op = tf.metrics.auc(
                        labels=g.y, predictions=g.soft_preds_scalar)
                    g.train_precision_val, g.train_precision_op = tf.metrics.precision(
                        labels=g.y, predictions=g.hard_preds_scalar)
                    g.train_recall_val, g.train_recall_op = tf.metrics.recall(
                        labels=g.y, predictions=g.hard_preds_scalar)

                    g.val_acc_val, g.val_acc_op = tf.metrics.accuracy(
                        labels=g.y, predictions=g.hard_preds_scalar)
                    g.val_auc_val, g.val_auc_op = tf.metrics.auc(
                        labels=g.y, predictions=g.soft_preds_scalar)
                    g.val_precision_val, g.val_precision_op = tf.metrics.precision(
                        labels=g.y, predictions=g.hard_preds_scalar)
                    g.val_recall_val, g.val_recall_op = tf.metrics.recall(
                        labels=g.y, predictions=g.hard_preds_scalar)

            # EXPERIMENTAL: tf.metrics 'mean_per_class_accuracy', 'precision_at_k',
            # and 'recall_at_k' for multi- classification
            k = 1  # top-1 scores
            if self.n_outputs > 2:
                with tf.name_scope('multiclass_metrics'):
                    g.train_acc_val, g.train_acc_op = tf.metrics.mean_per_class_accuracy(
                        g.y, g.hard_preds_scalar, num_classes=self.n_outputs)
                    g.train_precision_val, g.train_precision_op = tf.metrics.precision_at_k(
                        g.y_2d, g.hard_preds_sparse, k)
                    g.train_recall_val, g.train_recall_op = tf.metrics.recall_at_k(
                        g.y_2d, g.hard_preds_sparse, k)

                    g.val_acc_val, g.val_acc_op = tf.metrics.mean_per_class_accuracy(
                        g.y, g.hard_preds_scalar, num_classes=self.n_outputs)
                    g.val_precision_val, g.val_precision_op = tf.metrics.precision_at_k(
                        g.y_2d, g.hard_preds_sparse, k)
                    g.val_recall_val, g.val_recall_op = tf.metrics.recall_at_k(
                        g.y_2d, g.hard_preds_sparse, k)

            with tf.name_scope('train'):
                g.optimizer = self.optimizer(learning_rate=float(self.params['eta']))
                g.training_op = g.optimizer.minimize(g.combined_loss)

            g.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.name_scope('xentropy'):
                g.train_xentropy = tf.summary.scalar('train', g.xentropy_mean)
                g.val_xentropy = tf.summary.scalar('val', g.xentropy_mean)

            if self.params['regularizer'] != 'None':
                with tf.name_scope('reg_losses'):
                    g.train_reg_loss = tf.summary.scalar('train', tf.add_n(g.reg_losses))
                    g.val_reg_loss = tf.summary.scalar('val', tf.add_n(g.reg_losses))

            with tf.name_scope('ROC_AUC'):
                g.train_auc = tf.summary.scalar('train', g.train_auc_val)
                g.val_auc = tf.summary.scalar('val', g.val_auc_val)

            with tf.name_scope('accuracy'):
                g.train_acc = tf.summary.scalar('train', g.train_acc_val)
                g.val_acc = tf.summary.scalar('val', g.val_acc_val)

            g.file_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())

            g.saver = tf.train.Saver()

            return g

    def train_eval(self, X_train, y_train, X_val, y_val, return_preds=False,
                    save_ckpt=False):

        g = self.build_graph()

        self.evals_out = {'round': [],
                          'train': {'AUC': [], 'acc': [], 'precision': [], 'recall': []},
                          'val': {'AUC': [], 'acc': [], 'precision': [], 'recall': []}}

        train_batch_size = self.params['train_batch_size']
        val_batch_size = self.params['val_batch_size']
        n_val_batches = self.params['n_val_batches']

        if not return_preds:
            self.logger.info(f'  RND              TRAIN              |               VAL')
            self.logger.info(f'         acc     auc    prec   recall |    acc     auc    prec   recall')

        self.sess = tf.InteractiveSession(graph=g.graph,
                            config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

        if self.params['train_mode'] == 'minibatch':
            n_train_batches = (X_train.shape[0] // train_batch_size) + 1
            train_batch_size = X_train.shape[0] // n_train_batches
            self.logger.info(f'CV batch size scaled: {train_batch_size} n_train_batches {n_train_batches}')
            self.params['tboard_evals_step'] = 1
            self.params['log_evals_step'] = 1
            self.logger.info(f'evals set to every epoch')
        for epoch in range(self.params['n_epochs']):
            if self.params['train_mode'] == 'minibatch':
                idx = self._shuffle_idx(X_train)
                for batch in range(1, n_batches+1):
                    X_train_batch, y_train_batch = self.get_batch(X_train,
                                        y_train, idx, train_batch_size, batch)
                    train_op_dict = {g.X: X_train_batch, g.y: y_train_batch, g.train_flag:True}
                    self.sess.run([g.training_op, g.extra_update_ops],
                                   feed_dict=train_op_dict)

            elif self.params['train_mode'] == 'sgd':
                X_train_batch, y_train_batch = self.get_sample(
                    X_train, y_train, batch_size)
                train_op_dict = {X: X_train_batch,
                                 y: y_train_batch,
                                 train_flag:True}
                self.sess.run([g.training_op, g.extra_update_ops],
                               feed_dict=train_op_dict)

            # Tensorboard evals
            if ((epoch + 1) % self.params['tboard_evals_step'] == 0
                and not return_preds):

                self.sess.run(tf.local_variables_initializer())

                train_eval_dict = {g.X: X_train_batch, g.y: y_train_batch, g.train_flag:False}
                self.sess.run(g.train_acc_op, feed_dict=train_eval_dict)

                train_xent_summ, train_acc_summ =\
                    self.sess.run([g.train_xentropy, g.train_acc],
                    feed_dict=train_eval_dict)
                g.file_writer.add_summary(train_xent_summ, epoch+1)
                g.file_writer.add_summary(train_acc_summ, epoch+1)

                X_val_batch, y_val_batch = self.get_sample(X_val, y_val, val_batch_size)
                val_eval_dict = {g.X: X_val_batch, g.y: y_val_batch, g.train_flag:False}
                self.sess.run(g.val_acc_op, feed_dict=val_eval_dict)

                val_xent_summ, val_acc_summ =\
                    self.sess.run([g.val_xentropy, g.val_acc],
                    feed_dict=val_eval_dict)
                g.file_writer.add_summary(val_xent_summ, epoch+1)
                g.file_writer.add_summary(val_acc_summ, epoch+1)

                # eval AUC for binary classification only
                if self.n_outputs == 1:
                    self.sess.run(g.train_auc_op, feed_dict=train_eval_dict)
                    train_auc_summ = self.sess.run(g.train_auc, feed_dict=train_eval_dict)
                    g.file_writer.add_summary(train_auc_summ, epoch+1)

                    self.sess.run(g.val_auc_op, feed_dict=train_eval_dict)
                    val_auc_summ = self.sess.run(g.val_auc, feed_dict=val_eval_dict)
                    g.file_writer.add_summary(val_auc_summ, epoch+1)

                if self.params['regularizer'] in ['l1', 'l2', 'l1-l2']:
                    train_reg_loss_summ = g.train_reg_loss.eval(
                        feed_dict=train_eval_dict)
                    g.file_writer.add_summary(train_reg_loss_summ, epoch)
                    val_reg_loss_summ = g.val_reg_loss.eval(
                        feed_dict=val_eval_dict)
                    g.file_writer.add_summary(val_reg_loss_summ, epoch)

            # logger evals for BINARY classification
            if ((epoch + 1) % self.params['log_evals_step'] == 0
                and self.n_outputs == 1 and not return_preds):

                self.sess.run(tf.local_variables_initializer())

                for i in range(n_val_batches):
                    X_train_batch, y_train_batch = self.get_sample(
                                            X_train, y_train, train_batch_size)

                    self.sess.run([g.train_acc_op, g.train_auc_op, g.train_precision_op, g.train_recall_op],
                        feed_dict={g.X: X_train_batch, g.y: y_train_batch, g.train_flag:False})

                    X_val_batch, y_val_batch = self.get_sample(
                                            X_val, y_val, val_batch_size)

                    self.sess.run([g.val_acc_op, g.val_auc_op, g.val_precision_op, g.val_recall_op],
                        feed_dict={g.X: X_val_batch, g.y: y_val_batch, g.train_flag:False})

                train_acc_, train_auc_, train_precision_, train_recall_  =\
                    self.sess.run([g.train_acc_val, g.train_auc_val, g.train_precision_val, g.train_recall_val])

                val_acc_, val_auc_, val_precision_, val_recall_  =\
                    self.sess.run([g.val_acc_val, g.val_auc_val, g.val_precision_val, g.val_recall_val])

                # log evals
                self.logger.info(f' {str(epoch + 1):>4}  {train_acc_:.4f}  {train_auc_:.4f}  ' +
                                 f'{train_precision_:.4f}  {train_recall_:.4f} |  ' +
                                 f'{val_acc_:.4f}  {val_auc_:.4f}  ' +
                                 f'{val_precision_:.4f}  {val_recall_:.4f}')

                # record evals to self.evals_out (for plot_results)
                self.evals_out['round'].append(epoch + 1)
                self.evals_out['train']['acc'].append(train_acc_)
                self.evals_out['train']['AUC'].append(train_auc_)
                self.evals_out['train']['precision'].append(train_precision_)
                self.evals_out['train']['recall'].append(train_recall_)
                self.evals_out['val']['acc'].append(val_acc_)
                self.evals_out['val']['AUC'].append(val_auc_)
                self.evals_out['val']['precision'].append(val_precision_)
                self.evals_out['val']['recall'].append(val_recall_)

            # logger evals for MULTICLASS  classification
            if ((epoch + 1) % self.params['log_evals_step'] == 0
                and self.n_outputs > 2 and not return_preds):

                self.sess.run(tf.local_variables_initializer())

                for i in range(n_val_batches):
                    X_train_batch, y_train_batch = self.get_sample(
                                            X_train, y_train, train_batch_size)

                    self.sess.run([g.train_acc_op, g.train_precision_op, g.train_recall_op],
                        feed_dict={g.X: X_train_batch, g.y: y_train_batch, g.train_flag:False})

                    X_val_batch, y_val_batch = self.get_sample(
                                            X_val, y_val, val_batch_size)

                    self.sess.run([g.val_acc_op, g.val_precision_op, g.val_recall_op],
                        feed_dict={g.X: X_val_batch, g.y: y_val_batch, g.train_flag:False})

                train_acc_, train_precision_, train_recall_  =\
                    self.sess.run([g.train_acc_val, g.train_precision_val, g.train_recall_val])

                val_acc_, val_precision_, val_recall_  =\
                    self.sess.run([g.val_acc_val, g.val_precision_val, g.val_recall_val])

                # log evals
                self.logger.info(f' {str(epoch + 1):>4}  {train_acc_:.4f}          ' +
                                 f'{train_precision_:.4f}  {train_recall_:.4f} |  ' +
                                 f'{val_acc_:.4f}          ' +
                                 f'{val_precision_:.4f}  {val_recall_:.4f}')

                # record evals for plot_results()
                self.evals_out['round'].append(epoch + 1)
                self.evals_out['train']['acc'].append(train_acc_)
                self.evals_out['train']['precision'].append(train_precision_)
                self.evals_out['train']['recall'].append(train_recall_)
                self.evals_out['val']['acc'].append(val_acc_)
                self.evals_out['val']['precision'].append(val_precision_)
                self.evals_out['val']['recall'].append(val_recall_)

        if save_ckpt:
            save_path = saver.save(self.sess, self.ckpt_file)
            file_writer.close()
            self.logger.info(f'checkpoint saved as \'{self.ckpt_file}\'.')

        #------- TODO: ADD SUPPORT FOR MULTI-CLASS -------
        if return_preds and self.n_outputs == 1:
            chunk_size = int(self.params['predict_chunk_size'])
            n_chunks = X_val.shape[0] // chunk_size + 1
            fold_preds = []
            for i in range(n_chunks):
                feed_dict={train_flag:False,
                    X: X_val.iloc[(i*chunk_size):((i+1)*chunk_size), :].values}
                preds_chunk = g.soft_preds_scalar.eval(feed_dict=feed_dict)
                fold_preds.extend(preds_chunk.ravel())
            return fold_preds
