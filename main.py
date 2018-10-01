import argparse
import os
import tensorflow as tf
from model import Model
from configure import conf


"""This script defines hyperparameters.
"""


def main(_):
	parser = argparse.ArgumentParser()
	parser.add_argument('--option', dest='option', type=str, default='train',
						help='actions: train or predict')
	args = parser.parse_args()

	if args.option not in ['train', 'predict']:
		print('invalid option: ', args.option)
		print("Please input a option: train or predict")
	else:
		model = Model(conf)
		getattr(model, args.option)()


if __name__ == '__main__':
	# Choose which gpu or cpu to use
	os.environ['CUDA_VISIBLE_DEVICES'] = '5'
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()
