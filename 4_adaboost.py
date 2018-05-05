
import pandas as pd
import datetime
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import h5py
import os
from datetime import datetime
from functools import partial


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
agg1.to_csv('output_3/srch_dest_hc_hm_agg.csv', index=False)
del agg

destinations = pd.read_csv('destinations.csv')
submission = pd.read_csv('sample_submission.csv')


clf = AdaBoostClassifier(n_estimators=10)


tchunksize = 1000000
train = pd.read_csv('train.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], skiprows=0, nrows=chunksize)
train = train[train.is_booking==1]
train = pd.merge(train, destinations, how='left', on='srch_destination_id')
train = pd.merge(train, agg1, how='left', on=['srch_destination_id','hotel_country','hotel_market'])
pre_process(train)
y = train.hotel_cluster
train.drop(['cnt', 'hotel_cluster', 'is_booking'], axis=1, inplace=True)


clf.fit(train, y)


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

with h5py.File('output_3/probs/allpreds_xgb.h5', 'w') as hf:
    print('writing latest probabilities to file')
    hf.create_dataset('preds', data=preds)

print('generating submission')
col_ind = np.argsort(-preds, axis=1)[:,:5]
hc = [' '.join(row.astype(str)) for row in col_ind]

sub = pd.DataFrame(data=hc, index=submission.id)
sub.reset_index(inplace=True)
sub.columns = submission.columns
sub.to_csv('output_3/pred_sub.csv', index=False)
