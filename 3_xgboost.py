import pandas as pd
import datetime
from sklearn import cross_validation
import xgboost as xgb
import numpy as np
import h5py
import os
from datetime import datetime
from functools import partial


#using map5eval matrix for cross-validation
#this metric is also used for final testing as given by kaggle
def map5eval(preds, dtrain):
    actual = dtrain.get_label()
    predicted = preds.argsort(axis=1)[:,-np.arange(5)]
    metric = 0.
    for i in range(5):
        metric += np.sum(actual==predicted[:,i])/(i+1)
    metric /= actual.shape[0]
    return 'MAP@5', -metric

def pre_process(data):
    to_datetime_fmt = partial(pd.to_datetime, format='%Y-%m-%d %H:%M:%S')
    try:
        data.loc[data.srch_ci.str.endswith('00'),'srch_ci'] = '2015-12-31'
        data['srch_ci'] = data.srch_ci.astype(np.datetime64)
        data.loc[data.date_time.str.endswith('00'),'date_time'] = '2015-12-31'
        data['date_time'] = data.date_time.astype(np.datetime64)
    except:
        pass
    data.fillna(0, inplace=True)
    data.srch_ci = data.srch_ci.apply(to_datetime_fmt)
    data.srch_co = data.srch_co.apply(to_datetime_fmt)
    data['srch_duration'] = data.srch_co-data.srch_ci
    data['srch_duration'] = pd.to_timedelta(data['srch_duration'])
    data['srch_duration'] = data['srch_duration']/np.timedelta64(1, 'D')   
    data.srch_ci = data.srch_ci.apply(to_datetime_fmt)
    data['time_to_ci'] = data.srch_ci-data.date_time
    data['time_to_ci'] = pd.to_timedelta(data['time_to_ci'])/np.timedelta64(1, 'D')
    data['ci_month'] = data['srch_ci'].apply(lambda dt: dt.month)
    data['ci_day'] = data['srch_ci'].apply(lambda dt: dt.day)
    data['bk_month'] = data['date_time'].apply(lambda dt: dt.month)
    data['bk_day'] = data['date_time'].apply(lambda dt: dt.day)
    data['bk_hour'] = data['date_time'].apply(lambda dt: dt.hour)
    data.drop(['date_time', 'user_id', 'srch_ci', 'srch_co'], axis=1, inplace=True)



reader = pd.read_csv('train_new.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=200000)
pieces = [chunk.groupby(['srch_destination_id','hotel_country','hotel_market','hotel_cluster'])['is_booking'].agg(['sum','count']) for chunk in reader]
agg = pd.concat(pieces).groupby(level=[0,1,2,3]).sum()
del pieces
agg.dropna(inplace=True)
agg['sum_and_cnt'] = 0.85*agg['sum'] + 0.15*agg['count']
agg = agg.groupby(level=[0,1,2]).apply(lambda x: x.astype(float)/x.sum())
agg.reset_index(inplace=True)
agg1 = agg.pivot_table(index=['srch_destination_id','hotel_country','hotel_market'], columns='hotel_cluster', values='sum_and_cnt').reset_index()
agg1.to_csv('output_2/srch_dest_hc_hm_agg.csv', index=False)
del agg

destinations = pd.read_csv('destinations.csv')
submission = pd.read_csv('sample_submission.csv')

clf = xgb.XGBClassifier(objective = 'multi:softmax',
                max_depth = 5,
                n_estimators=300,
                learning_rate=0.01,
                nthread=4,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight = 3,
                silent=False)


#taking whole training data as chunk
chunksize = 30136234
train = pd.read_csv('train_new.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], skiprows=0, nrows=chunksize)
#considering only booking data
train = train[train.is_booking==1]
train = pd.merge(train, destinations, how='left', on='srch_destination_id')
train = pd.merge(train, agg1, how='left', on=['srch_destination_id','hotel_country','hotel_market'])
pre_process(train)
y = train.hotel_cluster
train.drop(['cnt', 'hotel_cluster', 'is_booking'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, y, stratify=y, test_size=0.2)
# cross validation
clf.fit(X_train, y_train, early_stopping_rounds=1, eval_metric=map5eval, eval_set=[(X_train, y_train),(X_test, y_test)])


count = 0
chunksize = 10000
preds = np.empty((submission.shape[0],clf.n_classes_))
reader = pd.read_csv('test.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=chunksize)
for chunk in reader:
    chunk = pd.merge(chunk, destinations, how='left', on='srch_destination_id')
    chunk = pd.merge(chunk, agg1, how='left', on=['srch_destination_id','hotel_country','hotel_market'])
    chunk.drop(['id'], axis=1, inplace=True)
    pre_process(chunk)
    
    pred = clf.predict_proba(chunk)
    preds[count:(count + chunk.shape[0]),:] = pred
    count = count + chunksize
    print('%d rows completed' % count)

del clf
del agg1
#allpreds_xgb.hs contains all the probabilities
with h5py.File('output_2/probs/allpreds_xgb.h5', 'w') as hf:
    print('writing latest probabilities to file')
    hf.create_dataset('preds', data=preds)

print('generating submission')
col_ind = np.argsort(-preds, axis=1)[:,:5]
hc = [' '.join(row.astype(str)) for row in col_ind]

sub = pd.DataFrame(data=hc, index=submission.id)
sub.reset_index(inplace=True)
sub.columns = submission.columns
sub.to_csv('output_2/pred_sub.csv', index=False)
