import numpy as np
import os
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from utils import load_subject, dice_ratio, ModHausdorffDist
import scipy.io as sio


"""Perform evaluation in terms of dice ratio and 3D MHD.
"""


################################################################################
# Arguments
################################################################################
RAW_DATA_DIR = '/tempspace2/zwang6/InfantBrain/RawData'
LABEL_DIR = '/tempspace2/zwang6/InfantBrain/tfrecords'
PRED_DIR = './results'
CHECKPOINT_NUM = 153000
VALID_ID = 10
OVERLAP_STEPSIZE = 8
VISUALE_DEPTH = 150


################################################################################
# Functions
################################################################################
def one_hot(label):
	'''Convert label (d,h,w) to one-hot label (d,h,w,num_class).
	'''

	num_class = np.max(label) + 1
	return np.eye(num_class)[label]


def MHD_3D(pred, label):
	'''Compute 3D MHD for a single class.
	
	Args:
		pred: An array of size [Depth, Height, Width], with only 0 or 1 values
		label: An array of size [Depth, Height, Width], with only 0 or 1 values

	Returns:
		3D MHD for a single class
	'''

	D, H, W = label.shape

	pred_d = np.array([pred[:, i, j] for i in range(H) for j in range(W)])
	pred_h = np.array([pred[i, :, j] for i in range(D) for j in range(W)])
	pred_w = np.array([pred[i, j, :] for i in range(D) for j in range(H)])

	label_d = np.array([label[:, i, j] for i in range(H) for j in range(W)])
	label_h = np.array([label[i, :, j] for i in range(D) for j in range(W)])
	label_w = np.array([label[i, j, :] for i in range(D) for j in range(H)])

	MHD_d = ModHausdorffDist(pred_d, label_d)[0]
	MHD_h = ModHausdorffDist(pred_h, label_h)[0]
	MHD_w = ModHausdorffDist(pred_w, label_w)[0]

	ret = np.mean([MHD_d, MHD_h, MHD_w])

	print('--->MHD d:', MHD_d)
	print('--->MHD h:', MHD_h)
	print('--->MHD w:', MHD_w)
	# print('--->avg:', ret)

	return ret


def Postprocess(input_T1, pred):
	pass


# def Visualize(pred, label, checkpoint_num, valid_id, overlap_step, depth):
# 	pred_show = pred[:, :, depth]
# 	label_show = label[:, :, depth]

# 	fig = plt.figure()
# 	fig.suptitle('Compare the %d-th slice.' % depth, fontsize=14)

# 	a = fig.add_subplot(1,2,1)
# 	imgplot = plt.imshow(label_show)
# 	a.set_title('Groud Truth')

# 	a = fig.add_subplot(1,2,2)
# 	imgplot = plt.imshow(pred_show)
# 	a.set_title('Prediction')

# 	plt.savefig('visualization-%d-sub-%d-overlap-%d' % (checkpoint_num, valid_id, overlap_step))


def Evaluate(label_dir, pred_dir, checkpoint_num, valid_id,
				overlap_step, post_process_fn, visualize_fn):
	print('Perform evaluation for subject-%d:' % valid_id)

	print('Loading label...')
	label_file = os.path.join(label_dir, 
				'subject-%d-label.npy' % valid_id)
	assert os.path.isfile(label_file), \
			('Run utils/generate_tfrecord.py to generate the corresponding label file.')
	label = np.load(label_file)
	print('Check label: ', label.shape, np.max(label))

	print('Loading predition...')
	pred_file = os.path.join(pred_dir, 
				'preds-%d-sub-%d-overlap-%d.npy' % (checkpoint_num, valid_id, overlap_step))
	assert os.path.isfile(pred_file), \
			('Run main.py --option=predict to generate the corresponding prediction file.')
	pred = np.load(pred_file)

	if post_process_fn != None:
		[T1, _, _] = load_subject(RAW_DATA_DIR, valid_id)
		pred = post_process_fn(T1, pred)

	print('Check pred: ', pred.shape, np.max(pred))

	if visualize_fn != None:
		print('Generating the visualization...')
		visualize_fn(pred, label, checkpoint_num, valid_id, overlap_step, VISUALE_DEPTH)

	print('Extract pred and label for each class...')
	label_one_hot = one_hot(label)
	pred_one_hot = one_hot(pred)
	print('Check shape: ', label_one_hot.shape, pred_one_hot.shape)

	# Separate each class. 0 corresponds to the background class (ignore).
	csf_pred = pred_one_hot[:,:,:,1]
	csf_label = label_one_hot[:,:,:,1]

	gm_pred = pred_one_hot[:,:,:,2]
	gm_label = label_one_hot[:,:,:,2]

	wm_pred = pred_one_hot[:,:,:,3]
	wm_label = label_one_hot[:,:,:,3]

	sio.savemat("our_model.mat",
		{"pred": pred[:, :, VISUALE_DEPTH],
		"csf": csf_pred[:, :, VISUALE_DEPTH],
		"gm": gm_pred[:, :, VISUALE_DEPTH],
		"wm": wm_pred[:, :, VISUALE_DEPTH]})
	sio.savemat("ground_truth.mat",
		{"pred": label[:, :, VISUALE_DEPTH],
		"csf": csf_label[:, :, VISUALE_DEPTH],
		"gm": gm_label[:, :, VISUALE_DEPTH],
		"wm": wm_label[:, :, VISUALE_DEPTH]})

	# # evaluate dice ratio
	# print('Evaluate dice ratio...')
	# csf_dr = dice_ratio(csf_pred, csf_label)
	# print('--->CSF Dice Ratio:', csf_dr)
	# gm_dr = dice_ratio(gm_pred, gm_label)
	# print('--->GM Dice Ratio:', gm_dr)
	# wm_dr = dice_ratio(wm_pred, wm_label)
	# print('--->WM Dice Ratio:', wm_dr)
	# print('--->avg:', np.mean([csf_dr, gm_dr, wm_dr]))

	# # evaluate MHD
	# print('Evaluate 3D MHD (---SLOW---)...')
	# csf_mhd = MHD_3D(csf_pred, csf_label)
	# print('--->CSF MHD:', csf_mhd)
	# gm_mhd = MHD_3D(gm_pred, gm_label)
	# print('--->GM MHD:', gm_mhd)
	# wm_mhd = MHD_3D(wm_pred, wm_label)
	# print('--->WM MHD:', wm_mhd)
	# print('--->avg:', np.mean([csf_mhd, gm_mhd, wm_mhd]))

	print('Done.')

if __name__ == '__main__':
	Evaluate(
		label_dir=LABEL_DIR,
		pred_dir=PRED_DIR,
		checkpoint_num=CHECKPOINT_NUM,
		valid_id=VALID_ID,
		overlap_step=OVERLAP_STEPSIZE,
		post_process_fn=None,
		visualize_fn=None)