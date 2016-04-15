import sys
sys.path.append("/cal/homes/jschoumacher/workspace/sd/Python/scikit-learn")

import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier,VotingClassifier


data1 = pandas.read_csv("floatCategorizedTest.csv",sep=";")
data2 = pandas.read_csv("floatCategorized.csv",sep=";")


cols = ['VOIE_DEPOT','SOURCE_BEGIN_MONTH','APP_NB',
       'APP_NB_PAYS', 'APP_NB_TYPE', 'FIRST_CLASSE',
       'NB_CLASSES', 'NB_ROOT_CLASSES', 'NB_SECTORS', 'NB_FIELDS',
       'MAIN_IPC', 'INV_NB', 'INV_NB_PAYS', 'INV_NB_TYPE',
       'cited_age_mean','SOURCE_CITED_AGE',
       'NB_BACKWARD_NPL', 'NB_BACKWARD_XY','NB_BACKWARD', 'pct_NB_IPC','oecd_NB_ROOT_CLASSES','oecd_NB_BACKWARD_PL',
       'IDX_ORIGIN', 'SOURCE_IDX_ORI', 'IDX_RADIC',
       'SOURCE_IDX_RAD', 'PUBLICATION_MONTH']

for i in range(1,92):
    cols.append('COUNTRY_'+str(i))

for i in range(1,129):
    cols.append('FISRT_APP_COUNTRY_'+str(i))

for i in range(1,5):
    cols.append('FISRT_APP_TYPE_'+str(i))

for i in range(1,30):
    cols.append('LANGUAGE_OF_FILLING_'+str(i))

for i in range(1,6):
    cols.append('TECHNOLOGIE_SECTOR_'+str(i))

for i in range(1,36):
    cols.append('TECHNOLOGIE_FIELD_'+str(i))

for i in range(1,137):
    cols.append('FISRT_INV_COUNTRY_'+str(i))

for i in range(1,6):
    cols.append('FISRT_INV_TYPE_'+str(i))

print(cols)
print(len(cols))

X1 = data1[cols]
#y1 = data1.VARIABLE_CIBLE_1
X2 = data2[cols]
y2 = data2.VARIABLE_CIBLE_1

scaler = preprocessing.StandardScaler().fit(X1)
X1 = scaler.transform(X1)
scaler = preprocessing.StandardScaler().fit(X2)
X2 = scaler.transform(X2)

#X1,y1 = shuffle(X1,y1)
X2,y2 = shuffle(X2,y2)

print(X1.shape)
print(X2.shape)

def createVotingClassifier(n_trees,X,y,depth,min_saples=2,max_feat=0.2,overhead=2.0,voting_='soft'):
    N_data = int(overhead*len(X)/n_trees)
    print(str(N_data)+' will be used by classifier')
    estimators_ = []
    estimators = []
    for i in range(n_trees):
        clf = RandomForestClassifier(max_depth=depth,min_samples_leaf=min_saples,max_features=max_feat)
	if (i+1)*N_data<len(X):
        	clf.fit(X[i*N_data:(i+1)*N_data],y[i*N_data:(i+1)*N_data])
	else:
		X,y = shuffle(X,y)
		clf.fit(X[:N_data],y[:N_data])
        estimators_.append((str(i),clf))
        estimators.append(clf)
    tmp = VotingClassifier(estimators=estimators_, voting=voting_)
    tmp.estimators_ = estimators
    return tmp

M = 200000

subdata1 = X2[:M]
subdata2 = X2[M:]
y_sub1 = y2[:M]
y_sub2 = y2[M:]

curN = 14
curN_leaf_min = 2
curDepth = 55
curRatio_filling = 0.13
curOverheap = 4.5

boundsN = (5,30)
boundsN_leaf_min = (1,5)
boundsDepth = (20,100)
boundsRatio_filling = (0.01,0.4,0.02)
boundsOverheap = (1.0,6.0,0.25)

lengthDepth = boundsDepth[1]-boundsDepth[0]
lengthRatio_filling = boundsRatio_filling[1]-boundsRatio_filling[0]
lengthOverheap = boundsOverheap[1]-boundsOverheap[0]

maxScore = 0
n = 0
while n<20:
    print(curN,curN_leaf_min,curDepth,curRatio_filling,curOverheap," iterating on curRatio")
    cur_maxScore = 0
    if curRatio_filling-lengthRatio_filling>=0.01:
        x = np.arange(curRatio_filling-lengthRatio_filling,curRatio_filling+lengthRatio_filling,boundsRatio_filling[2])
    else:
        x = np.arange(0.01,curRatio_filling+lengthRatio_filling,boundsRatio_filling[2])
    for i in x:
        myClassifier = createVotingClassifier(curN,subdata1,y_sub1,curDepth,curN_leaf_min,i,curOverheap)
        y2_pred = myClassifier.predict_proba(subdata2)
        score = roc_auc_score(y_sub2.values,y2_pred[:,0])
        print("curRatio = ",i," => ",score)
        if score>cur_maxScore:
            cur_maxScore = score
            curRatio_filling = i

    if cur_maxScore>maxScore:
        if lengthRatio_filling/1.41>2*boundsRatio_filling[2]:
            lengthRatio_filling = lengthRatio_filling/1.41
        print("Score improved ! : ",maxScore," => ",cur_maxScore)
        maxScore = cur_maxScore
    else:
        print("Score not improved ! :'(' ")
        lengthRatio_filling = lengthRatio_filling*1.41

    print(curN,curN_leaf_min,curDepth,curRatio_filling,curOverheap," iterating on curOverheap")
    cur_maxScore = 0
    if curOverheap-lengthOverheap>=1.0:
        x = np.arange(curOverheap-lengthOverheap,curOverheap+lengthOverheap,boundsRatio_filling[2])
    else:
        x = np.arange(1.0,curOverheap+lengthOverheap,boundsRatio_filling[2])
    for i in x:
        myClassifier = createVotingClassifier(curN,subdata1,y_sub1,curDepth,curN_leaf_min,curRatio_filling,i)
        y2_pred = myClassifier.predict_proba(subdata2)
        score = roc_auc_score(y_sub2.values,y2_pred[:,0])
        print("curOverhead = ",i," => ",score)
        if score>cur_maxScore:
            cur_maxScore = score
            curOverheap = i

    if cur_maxScore>maxScore:
        if lengthOverheap/1.41>2*boundsOverheap[2]:
            lengthOverheap = lengthOverheap/1.41
        print("Score improved ! : ",maxScore," => ",cur_maxScore)
        maxScore = cur_maxScore
    else:
        print("Score not improved ! :'(' ")
        lengthOverheap = lengthOverheap*1.41

    print(curN,curN_leaf_min,curDepth,curRatio_filling,curOverheap," iterating on N_samples")
    cur_maxScore = 0
    if curDepth-lengthDepth>=20:
        x = range(curDepth-lengthDepth,curDepth+lengthDepth,5)
    else:
        x = range(20,curDepth+lengthDepth,5)
    for i in x:
        myClassifier = createVotingClassifier(curN,subdata1,y_sub1,i,curN_leaf_min,curRatio_filling,curOverheap)
        y2_pred = myClassifier.predict_proba(subdata2)
        score = roc_auc_score(y_sub2.values,y2_pred[:,0])
        print("curDepth = ",i," => ",score)
        if score>cur_maxScore:
            cur_maxScore = score
            curDepth = i

    if cur_maxScore>maxScore:
        if lengthDepth/1.41>10:
            lengthDepth = int(lengthDepth/1.41)
        print("Score improved ! : ",maxScore," => ",cur_maxScore)
        maxScore = cur_maxScore
    else:
        print("Score not improved ! :'(' ")
        lengthDepth = int(lengthDepth*1.41)


    print(curN,curN_leaf_min,curDepth,curRatio_filling,curOverheap," iterating on N_leaf_min")
    cur_maxScore = 0
    x = range(boundsN_leaf_min[0],boundsN_leaf_min[1])
    for i in x:
        myClassifier = createVotingClassifier(curN,subdata1,y_sub1,curDepth,i,curRatio_filling,curOverheap)
        y2_pred = myClassifier.predict_proba(subdata2)
        score = roc_auc_score(y_sub2.values,y2_pred[:,0])
        print("curN_leaf_min = ",i," => ",score)
        if score>cur_maxScore:
            cur_maxScore = score
            curN_leaf_min = i

    if cur_maxScore>maxScore:
        print("Score improved ! : ",maxScore," => ",cur_maxScore)
        maxScore = cur_maxScore
    else:
        print("Score not improved ! :'(' ")

    print(curN,curN_leaf_min,curDepth,curRatio_filling,curOverheap," iterating on N trees")
    cur_maxScore = 0
    if curN-lengthN>=5:
        x = range(curN-lengthN,curN+lengthN,1)
    else:
        x = range(5,curN+lengthN,1)
    for i in x:
        myClassifier = createVotingClassifier(i,subdata1,y_sub1,curDepth,curN_leaf_min,curRatio_filling,curOverheap)
        y2_pred = myClassifier.predict_proba(subdata2)
        score = roc_auc_score(y_sub2.values,y2_pred[:,0])
        print("curN = ",i," => ",score)
        if score>cur_maxScore:
            cur_maxScore = score
            curN = i

    if cur_maxScore>maxScore:
        if lengthN/1.41>2:
            lengthN = int(lengthN/1.41)
        print("Score improved ! : ",maxScore," => ",cur_maxScore)
        maxScore = cur_maxScore
    else:
        print("Score not improved ! :'(' ")
        lengthN = int(lengthN*1.41)

    n = n+1


# for i in x:
#     myClassifier = createVotingClassifier(14,subdata1,y_sub1,55,2,0.13,i)
#     y2_pred = myClassifier.predict_proba(subdata2)
#     print(i,roc_auc_score(y_sub2.values,y2_pred[:,0]))


#x = range(1,20)
#res = []
#for i in x:
#    myClassifier = createVotingClassifier(i,subdata1,y_sub1,75,2,0.13,1.5)
#    y2_pred = myClassifier.predict_proba(subdata2)
#    print(i,roc_auc_score(y_sub2.values,y2_pred[:,0]))
#    res.append(roc_auc_score(y_sub2.values,y2_pred[:,0]))

#plt.plot(x,res)


print(X1.shape)
print(X2.shape)
#myClassifier = createVotingClassifier(15,X2,y2,55,2,0.13,4.5)
myClassifier = createVotingClassifier(curN,subdata1,y_sub1,curN_samples,curN_leaf_min,curRatio_filling,curOverheap)
y_pred = myClassifier.predict_proba(X1)[:, 1]
np.savetxt('y_pred.txt', y_pred, fmt='%s')
