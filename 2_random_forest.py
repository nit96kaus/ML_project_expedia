
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import h5py
import os
from datetime import datetime
from functools import partial
#import matplotlib.pyplot as plt
#import seaborn as sns


def pre_process(data):
    to_datetime_fmt = partial(pd.to_datetime, format='%Y-%m-%d %H:%M:%S')
    try:
        #data contains some random dates with endswith "00" so making it '2015-12-31' 
        #according to kaggle this is the last date upto data is present
        data.loc[data.srch_ci.str.endswith('00'),'srch_ci'] = '2015-12-31'
        data['srch_ci'] = data.srch_ci.astype(np.datetime64)
        data.loc[data.date_time.str.endswith('00'),'date_time'] = '2015-12-31'
        data['date_time'] = data.date_time.astype(np.datetime64)
    except:
        pass
    #filling na values with 0
    data.fillna(0, inplace=True)
    data.srch_ci = data.srch_ci.apply(to_datetime_fmt)
    data.srch_co = data.srch_co.apply(to_datetime_fmt)
    #making new features
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
agg1.to_csv('output_1/srch_dest_hc_hm_agg.csv', index=False)
del agg

#destinations.csv contains more features 
destinations = pd.read_csv('destinations.csv')
submission = pd.read_csv('sample_submission.csv')

# draw correlation coefficient matrix
#corrmat = df.corr()
#plt.figure()
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(np.abs(corrmat) , vmax=.8, square=True)
#plt.yticks(rotation=0)
#plt.xticks(rotation=90)
#plt.show()
#plt.savefig('corr.png')


clf = RandomForestClassifier(n_estimators=0, n_jobs=-1, warm_start=True)
count = 0
chunksize = 200000

reader = pd.read_csv('train_new.csv', parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=chunksize)
for chunk in reader:
    try:
        # only taking the booking data, as test data contains only booking data not click data 
        chunk = chunk[chunk.is_booking==1]
        chunk = pd.merge(chunk, destinations, how='left', on='srch_destination_id')
        chunk = pd.merge(chunk, agg1, how='left', on=['srch_destination_id','hotel_country','hotel_market'])
        pre_process(chunk)
        y = chunk.hotel_cluster
        chunk.drop(['cnt', 'hotel_cluster', 'is_booking'], axis=1, inplace=True)
        #increasing n_estimators if finds unique hotel cluster in current chunk
        if len(y.unique()) == 100:
            print("checking...")
            clf.set_params(n_estimators=clf.n_estimators+1)
            clf.fit(chunk, y)

       
        count = count + chunksize
        print('%d rows completed' % count)
        if(count/chunksize == 300):
            break
    except Exception as e:
        print('Error: %s' % str(e))
        pass

count = 0
chunksize = 10000
# creating matrix for probabilities of 100 classes of each row
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
print('writing current probabilities to file')
#allpreds.h5 contains all probabilities corresponding to 100 hotel clusters
with h5py.File('output_1/probs/allpreds.h5', 'w') as hf:
    print('writing latest probabilities to file')
    hf.create_dataset('preds_latest', data=preds)


print('generating submission')
# kaggle wants submission in specific format
# format is each row in test data should have 5 clusters of hotel in 2nd column
# first column is "id"
# taking top 5 probabilities
col_ind = np.argsort(-preds, axis=1)[:,:5]
hc = [' '.join(row.astype(str)) for row in col_ind]

sub = pd.DataFrame(data=hc, index=submission.id)
sub.reset_index(inplace=True)
sub.columns = submission.columns
#Pred_sub.csv is predicted file for random forest model
sub.to_csv('output_1/pred_sub.csv', index=False)
