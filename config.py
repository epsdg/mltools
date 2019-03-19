
cfg = dict(

    default_n_processes=2,

    default_map_fn_backend='tpe',

    default_n_folds=5,

    logger_config=dict(

        version=1,
        formatters=dict(simple=dict(format='%(asctime)s: %(message)s',
                                    datefmt='%H:%M:%S')),
        handlers = {'console': {
                        'class': 'logging.StreamHandler',
                        'level': 'DEBUG',
                        'formatter' : 'simple',
                        'stream' : 'ext://sys.stdout'},
                        },
        loggers = {'mltools': dict(
            level =      'DEBUG',
            handlers =   ['console'],
            propagate =  False)},
        root=dict(
          level =       'DEBUG',
          handlers =    ['console']))

                )
