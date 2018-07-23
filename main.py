import argparse
import os
import tensorflow as tf
from model import Model


"""This script defines hyperparameters.
"""


def configure():
	flags = tf.app.flags

	# training
	flags.DEFINE_string('data_dir', '/tempspace2/zwang6/InfantBrain',
			'the directory where the input data is stored')
	flags.DEFINE_integer('num_training_subs', 9,
			'the number of subjects used for training')
	flags.DEFINE_integer('train_epochs', 100000,
			'the number of epochs to use for training')
	flags.DEFINE_integer('epochs_per_eval', 5000,
			'the number of training epochs to run between evaluations')
	flags.DEFINE_integer('batch_size', 5,
			'the number of examples processed in each training batch')
	flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
	flags.DEFINE_float('weight_decay', 2e-6, 'weight decay rate')
	flags.DEFINE_integer('num_parallel_calls', 5,
			'The number of records that are processed in parallel \
			during input processing. This can be optimized per data set but \
			for generally homogeneous data sets, should be approximately the \
			number of available CPU cores.')
	flags.DEFINE_string('model_dir', './model-1',
			'the directory where the model will be stored')

	# validation
	flags.DEFINE_integer('patch_size', 32, 'spatial size of patches')
	flags.DEFINE_integer('overlap_step', 8,
			'overlap step size when performing testing')
	flags.DEFINE_integer('validation_id', 1,
			'1-10, which subject is used for validation')
	flags.DEFINE_integer('checkpoint_num', 180000,
			'which checkpoint is used for validation')
	flags.DEFINE_string('save_dir', './results',
			'the directory where the prediction is stored')
	flags.DEFINE_string('raw_data_dir', '/tempspace2/zwang6/InfantBrain/RawData',
			'the directory where the raw data is stored')

	# network
	flags.DEFINE_integer('network_depth', 3, 'the network depth')
	flags.DEFINE_integer('num_classes', 4, 'the number of classes')
	flags.DEFINE_integer('num_filters', 32,
			 'number of filters for initial_conv')
	
	flags.FLAGS.__dict__['__parsed'] = False
	return flags.FLAGS


def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--option', dest='option', type=str, default='train',
						help='actions: train or predict')
	args = parser.parse_args()

	if args.option not in ['train', 'predict']:
		print('invalid option: ', args.option)
		print("Please input a option: train or predict")
	else:
		model = Model(configure())
		getattr(model, args.option)()


if __name__ == '__main__':
	# Choose which gpu or cpu to use
	os.environ['CUDA_VISIBLE_DEVICES'] = '5'
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()
