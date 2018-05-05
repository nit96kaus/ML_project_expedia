
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import h5py
import os


submission = pd.read_csv('sample_submission.csv')

# giving weightages to probabilities from all the 3 models
# and normalizing it such that sum of prob = 1
with h5py.File('output_1/probs/allpreds.h5', 'r') as hf:
       predshf = hf['preds_latest']
       preds = 0.44*normalize(predshf.value, norm='l1', axis=1)

with h5py.File('output_2/probs/allpreds_xgb.h5', 'r') as hf:
        predshf = hf['preds']
        preds = 0.55*normalize(predshf.value, norm='l1', axis=1)

with h5py.File('output_3/probs/allpreds_xgb.h5', 'r') as hf:
        predshf = hf['preds']
        preds += 0.01*normalize(predshf.value, norm='l1', axis=1)


print('generating submission')
col_ind = np.argsort(-preds, axis=1)[:,:5]
hc = [' '.join(row.astype(str)) for row in col_ind]

sub = pd.DataFrame(data=hc, index=submission.id)
sub.reset_index(inplace=True)
sub.columns = submission.columns
sub.to_csv('output/pred_sub.csv', index=False)
