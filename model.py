import tensorflow as tf
import os
import sys
import numpy as np

from network import Network
from input_fn import input_function
from generate_tfrecord import cut_edge, prepare_validation, load_subject


"""This script trains or evaluates the model.
"""


class Model(object):

	def __init__(self, conf):
		self.conf = conf


	def _model_fn(self, features, labels, mode):
		"""Initializes the Model representing the model layers
		and uses that model to build the necessary EstimatorSpecs for
		the `mode` in question. For training, this means building losses,
		the optimizer, and the train op that get passed into the EstimatorSpec.
		For evaluation and prediction, the EstimatorSpec is returned without
		a train op, but with the necessary parameters for the given mode.

		Args:
			features: tensor representing input images
			labels: tensor representing class labels for all input images
			mode: current estimator mode; should be one of
				`tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`

		Returns:
			ModelFnOps
		"""
		net = Network(self.conf)
		logits = net(features, mode == tf.estimator.ModeKeys.TRAIN)

		predictions = {
			'classes': tf.argmax(logits, axis=-1),
			'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
		}

		if mode == tf.estimator.ModeKeys.PREDICT:
			return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

		# Calculate loss, which includes softmax cross entropy and L2 regularization.
		cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
											labels=labels, logits=logits))

		# Create a tensor named cross_entropy for logging purposes.
		tf.identity(cross_entropy, name='cross_entropy')
		tf.summary.scalar('cross_entropy', cross_entropy)

		# Add weight decay to the loss.
		loss = cross_entropy + self.conf.weight_decay * tf.add_n(
				[tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name])

		if mode == tf.estimator.ModeKeys.TRAIN:
			global_step = tf.train.get_or_create_global_step()
			
			# Learning rate.
			# initial_learning_rate = self.conf.learning_rate
			# Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
			# boundaries = [int(batches_per_epoch * epoch) for epoch in [150, 200]]
			# vals = [initial_learning_rate * decay for decay in [1, 0.25, 0.25*0.25]]
			# learning_rate = tf.train.piecewise_constant(global_step, boundaries, vals)

			# Create a tensor named learning_rate for logging purposes
			# tf.identity(learning_rate, name='learning_rate')
			# tf.summary.scalar('learning_rate', learning_rate)

			# optimizer = tf.train.MomentumOptimizer(
			# 				learning_rate=learning_rate,
			# 				momentum=self.conf.momentum)

			optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.learning_rate)

			# Batch norm requires update ops to be added as a dependency to train_op
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = optimizer.minimize(loss, global_step)
		else:
			train_op = None

		accuracy = tf.metrics.accuracy(labels, predictions['classes'])
		metrics = {'accuracy': accuracy}

		# Create a tensor named train_accuracy for logging purposes
		tf.identity(accuracy[1], name='train_accuracy')
		tf.summary.scalar('train_accuracy', accuracy[1])

		return tf.estimator.EstimatorSpec(
				mode=mode,
				predictions=predictions,
				loss=loss,
				train_op=train_op,
				eval_metric_ops=metrics)


	def train(self):
		# Using the Winograd non-fused algorithms provides a small performance boost.
		os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

		save_checkpoints_steps = self.conf.epochs_per_eval * \
						self.conf.num_training_subs // self.conf.batch_size
		run_config = tf.estimator.RunConfig().replace(
						save_checkpoints_steps=save_checkpoints_steps,
						keep_checkpoint_max=0)

		classifier = tf.estimator.Estimator(
						model_fn=self._model_fn,
						model_dir=self.conf.model_dir,
						config=run_config)

		for _ in range(self.conf.train_epochs // self.conf.epochs_per_eval):
			tensors_to_log = {
				# 'learning_rate': 'learning_rate',
				'cross_entropy': 'cross_entropy',
				'train_accuracy': 'train_accuracy'
			}

			logging_hook = tf.train.LoggingTensorHook(
								tensors=tensors_to_log, every_n_iter=100)

			print('Starting a training cycle.')

			def input_fn_train():
				return input_function(
							data_dir=self.conf.data_dir,
							mode='train',
							patch_size=self.conf.patch_size,
							batch_size=self.conf.batch_size,
							buffer_size=self.conf.num_training_subs,
							valid_id=self.conf.validation_id,
							pred_id=-1, # not used
							overlap_step=-1, # not used
							num_epochs=self.conf.epochs_per_eval,
							num_parallel_calls=self.conf.num_parallel_calls)

			classifier.train(input_fn=input_fn_train, hooks=[logging_hook])

			if self.conf.validation_id != -1:
				print('Starting to evaluate.')

				def input_fn_eval():
					return input_function(
								data_dir=self.conf.data_dir,
								mode='valid',
								patch_size=self.conf.patch_size,
								batch_size=self.conf.batch_size,
								buffer_size=-1, # not used
								valid_id=self.conf.validation_id,
								pred_id=-1, # not used
								overlap_step=self.conf.overlap_step,
								num_epochs=1,
								num_parallel_calls=self.conf.num_parallel_calls)

				classifier.evaluate(input_fn=input_fn_eval)


	def predict(self):
		# Using the Winograd non-fused algorithms provides a small performance boost.
		os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

		print('Perform prediction for subject-%d:' % self.conf.prediction_id)

		print('Loading data...')
		[T1, _, _] = load_subject(self.conf.raw_data_dir, self.conf.prediction_id)

		(_, cut_size) = cut_edge(T1)
		print('Check cut_size: ',cut_size)

		cutted_T1 = T1[cut_size[0]:cut_size[1], cut_size[2]:cut_size[3], cut_size[4]:cut_size[5], :]
		patch_ids = prepare_validation(cutted_T1, self.conf.patch_size, self.conf.overlap_step)
		num_patches = len(patch_ids)
		print ('Number of patches:', num_patches)

		print('Initialize...')
		classifier = tf.estimator.Estimator(
						model_fn=self._model_fn,
						model_dir=self.conf.model_dir)

		def input_fn_predict():
			return input_function(
						data_dir=self.conf.data_dir,
						mode='pred',
						patch_size=self.conf.patch_size,
						batch_size=self.conf.batch_size,
						buffer_size=-1, # not used
						valid_id=-1, # not used
						pred_id=self.conf.prediction_id,
						overlap_step=self.conf.overlap_step,
						num_epochs=1,
						num_parallel_calls=self.conf.num_parallel_calls)

		checkpoint_file = os.path.join(self.conf.model_dir, 
							'model.ckpt-'+str(self.conf.checkpoint_num))

		preds = classifier.predict(
					input_fn=input_fn_predict,
					checkpoint_path=checkpoint_file)

		print('Starting to predict.')

		predictions = {}
		for i, pred in enumerate(preds):
			location = patch_ids[i]
			print('Step {:d}/{:d} processing results for ({:d},{:d},{:d})'.format(
						i+1, num_patches, location[0], location[1], location[2]),
						end='\r',
						flush=True)
			logits = pred['probabilities']
			for j in range(self.conf.patch_size):
				for k in range(self.conf.patch_size):
					for l in range(self.conf.patch_size):
						key = (location[0]+j, location[1]+k, location[2]+l)
						if key not in predictions.keys():
							predictions[key] = []
						predictions[key].append(logits[j, k, l, :])

		print('Averaging results...')

		results = np.zeros((T1.shape[0], T1.shape[1], T1.shape[2], self.conf.num_classes),
							dtype=np.float32)
		print(results.shape)
		for key in predictions.keys():
			results[cut_size[0]+key[0],	cut_size[2]+key[1], cut_size[4]+key[2]] = \
						np.mean(predictions[key], axis=0)
		results = np.argmax(results, axis=-1)

		print('Saving results...')

		if not os.path.exists(self.conf.save_dir):
			os.makedirs(self.conf.save_dir)
		save_filename = 'preds-' + str(self.conf.checkpoint_num) + \
						'-sub-' + str(self.conf.prediction_id) + \
						'-overlap-' + str(self.conf.overlap_step) + \
						'-patch-' + str(self.conf.patch_size) + '.npy'
		save_file = os.path.join(self.conf.save_dir, save_filename)
		np.save(save_file, results)

		print('Done.')

		os._exit(0)
