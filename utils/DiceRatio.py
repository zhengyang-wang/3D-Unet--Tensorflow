import numpy as np

def dice_ratio(pred, label):
    '''Note: pred & label should only contain 0 or 1.
    '''
    
    return np.sum(pred[label==1])*2.0 / (np.sum(pred) + np.sum(label))