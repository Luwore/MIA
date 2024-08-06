from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('arch', 'cnn32-3-max', 'Model architecture.')
flags.DEFINE_float('lr', 0.1, 'Learning rate.')
flags.DEFINE_string('dataset', 'cifar10', 'Dataset.')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay ratio.')
flags.DEFINE_float('momentum', 0.9, 'Momentum.')
flags.DEFINE_integer('batch', 64, 'Batch size')
flags.DEFINE_integer('epochs', 50, 'Training duration in number of epochs.')
flags.DEFINE_string('logdir', 'exp', 'Directory where to save checkpoints and tensorboard data.')
flags.DEFINE_integer('seed', None, 'Training seed.')
flags.DEFINE_float('pkeep', .5, 'Probability to keep examples.')
flags.DEFINE_integer('expid', 0, 'Experiment ID')  # Default value for expid
flags.DEFINE_integer('num_experiments', 16, 'Number of experiments')  # Default value for num_experiments
flags.DEFINE_string('augment', 'weak', 'Strong or weak augmentation')
flags.DEFINE_integer('only_subset', None, 'Only train on a subset of images.')
flags.DEFINE_integer('dataset_size', 50000, 'Number of examples to keep.')
flags.DEFINE_integer('eval_steps', 1, 'How often to get eval accuracy.')
flags.DEFINE_integer('abort_after_epoch', None, 'Stop training early at an epoch')
flags.DEFINE_integer('save_steps', 10, 'How often to save model.')
flags.DEFINE_integer('patience', None, 'Early stopping after this many epochs without progress')
flags.DEFINE_bool('tunename', False, 'Use tune name?')
flags.DEFINE_string('regex', '.*', 'Regex to filter files in the logdir.')