import os
import numpy as np
from utils import dice_ratio, ModHausdorffDist
from generate_tfrecord import load_subject


"""Perform evaluation in terms of dice ratio and 3D MHD.
"""


################################################################################
# Arguments
################################################################################
RAW_DATA_DIR = '/data/zhengyang/InfantBrain/RawData'
LABEL_DIR = '/data/zhengyang/InfantBrain/tfrecords_full'
PRED_DIR = './results'
PRED_ID = 10 # 1-10
PATCH_SIZE = 32
CHECKPOINT_NUM = 153000
OVERLAP_STEPSIZE = 8


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


def Evaluate(label_dir, pred_dir, pred_id, patch_size, checkpoint_num,
		overlap_step):
	print('Perform evaluation for subject-%d:' % pred_id)

	print('Loading label...')
	label_file = os.path.join(label_dir, 'subject-%d-label.npy' % pred_id)
	assert os.path.isfile(label_file), \
		('Run generate_tfrecord.py to generate the label file.')
	label = np.load(label_file)
	print('Check label: ', label.shape, np.max(label))

	print('Loading predition...')
	pred_file = os.path.join(pred_dir, 
				'preds-%d-sub-%d-overlap-%d-patch-%d.npy' % \
				(checkpoint_num, pred_id, overlap_step, patch_size))
	assert os.path.isfile(pred_file), \
		('Run main.py --option=predict to generate the prediction results.')
	pred = np.load(pred_file)
	print('Check pred: ', pred.shape, np.max(pred))

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

	# evaluate dice ratio
	print('Evaluate dice ratio...')
	csf_dr = dice_ratio(csf_pred, csf_label)
	print('--->CSF Dice Ratio:', csf_dr)
	gm_dr = dice_ratio(gm_pred, gm_label)
	print('--->GM Dice Ratio:', gm_dr)
	wm_dr = dice_ratio(wm_pred, wm_label)
	print('--->WM Dice Ratio:', wm_dr)
	print('--->avg:', np.mean([csf_dr, gm_dr, wm_dr]))

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
		pred_id=PRED_ID,
		patch_size=PATCH_SIZE,
		checkpoint_num=CHECKPOINT_NUM,
		overlap_step=OVERLAP_STEPSIZE)