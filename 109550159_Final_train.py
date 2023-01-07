import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, HuberRegressor
import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')
print(f'train {train.shape}, test {test.shape}')
print(f'failure 0: {train[train.failure==0].shape[0]}, failure 1: {train[train.failure==1].shape[0]}')

data = pd.concat([train, test])
data['m3_missing'] = data['measurement_3'].isnull().astype(np.int8)
data['m5_missing'] = data['measurement_5'].isnull().astype(np.int8)
data['loading'] = np.log1p(data['loading'])

feature = [f for f in test.columns if f.startswith('measurement') or f=='loading']

fill_dict = {
    'A': ['measurement_5','measurement_6','measurement_8'],
    'B': ['measurement_4','measurement_5','measurement_7'],
    'C': ['measurement_5','measurement_7','measurement_8','measurement_9'],
    'D': ['measurement_5','measurement_6','measurement_7','measurement_8'],
    'E': ['measurement_4','measurement_5','measurement_6','measurement_8'],
    'F': ['measurement_4','measurement_5','measurement_6','measurement_7'],
    'G': ['measurement_4','measurement_6','measurement_8','measurement_9'],
    'H': ['measurement_4','measurement_5','measurement_7','measurement_8','measurement_9'],
    'I': ['measurement_3','measurement_7','measurement_8']
}

for code in data.product_code.unique():
    tmp = data[data.product_code==code]
    column = fill_dict[code]
    tmp_train = tmp[column+['measurement_17']].dropna(how='any')
    tmp_test = tmp[(tmp[column].isnull().sum(axis=1)==0)&(tmp['measurement_17'].isnull())]
    print(f"code {code} has {len(tmp_test)} samples to fill nan")
    model = HuberRegressor()
    model.fit(tmp_train[column], tmp_train['measurement_17'])
    data.loc[(data.product_code==code)&(data[column].isnull().sum(axis=1)==0)&(data['measurement_17'].isnull()), 'measurement_17'] = model.predict(tmp_test[column])

    model2 = KNNImputer(n_neighbors=5)
    print(f"KNN imputing code {code}")
    data.loc[data.product_code==code, feature] = model2.fit_transform(data.loc[data.product_code==code, feature])

def _scale(train_data, val_data, test_data, feats):
    scaler = StandardScaler()
    # scaler = PowerTransformer()
    
    scaled_train = scaler.fit_transform(train_data[feats])
    scaled_val = scaler.transform(val_data[feats])
    scaled_test = scaler.transform(test_data[feats])
    
    #back to dataframe
    new_train = train_data.copy()
    new_val = val_data.copy()
    new_test = test_data.copy()
    
    new_train[feats] = scaled_train
    new_val[feats] = scaled_val
    new_test[feats] = scaled_test
    
    assert len(train_data) == len(new_train)
    assert len(val_data) == len(new_val)
    assert len(test_data) == len(new_test)
    
    return new_train, new_val, new_test

train = data[data.failure.notnull()]
test = data[data.failure.isnull()]
print(train.shape, test.shape)

X = train.drop(['failure'], axis=1)
y = train['failure'].astype(int)
test = test.drop(['failure'], axis=1)

select_feature = ['m3_missing', 'm5_missing', 'measurement_1', 'measurement_2', 'loading', 'measurement_17']

lr_oof_1 = np.zeros(len(train))
lr_oof_2 = np.zeros(len(train))
lr_test = np.zeros(len(test))
lr_auc = 0
lr_acc = 0
importance_list = []

#kf = GroupKFold(n_splits=5)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print("Fold:", fold_idx+1)
    x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    x_test = test.copy()
    
    x_train, x_val, x_test = _scale(x_train, x_val, x_test, select_feature)
    #x_train, x_val, x_test = x_train.round(2), x_val.round(2), x_test.round(2)
    
    model = LogisticRegression(max_iter=30000, C=0.00001, penalty='l2', solver='newton-cg') # , class_weight='balanced'
    model.fit(x_train[select_feature], y_train)
    importance_list.append(model.coef_.ravel())

    val_preds = model.predict_proba(x_val[select_feature])[:, 1]
    lr_auc += roc_auc_score(y_val, val_preds) / 5
    y_preds = model.predict(x_val[select_feature])
    lr_acc += accuracy_score(y_val, y_preds) / 5
    lr_test += model.predict_proba(x_test[select_feature])[:, 1] / 5
    lr_oof_1[val_idx] = val_preds
    lr_oof_2[val_idx] = y_preds

print(f"{Fore.GREEN}{Style.BRIGHT}Average auc = {round(lr_auc, 5)}, Average acc = {round(lr_acc, 5)}{Style.RESET_ALL}")
print(f"{Fore.RED}{Style.BRIGHT}OOF auc = {round(roc_auc_score(y, lr_oof_1), 5)}, OOF acc = {round(accuracy_score(y, lr_oof_2), 5)}{Style.RESET_ALL}")

submission['lr0'] = lr_test

select_feature = ['measurement_1', 'measurement_2', 'loading', 'measurement_17']

lr_oof_1 = np.zeros(len(train))
lr_oof_2 = np.zeros(len(train))
lr_test = np.zeros(len(test))
lr_auc = 0
lr_acc = 0
importance_list = []

#kf = GroupKFold(n_splits=5)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print("Fold:", fold_idx+1)
    x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    x_test = test.copy()
    
    x_train, x_val, x_test = _scale(x_train, x_val, x_test, select_feature)
    #x_train, x_val, x_test = x_train.round(2), x_val.round(2), x_test.round(2)
    
    model = LogisticRegression(max_iter=30000, C=0.00001, penalty='l2', solver='newton-cg') # , class_weight='balanced'
    model.fit(x_train[select_feature], y_train)
    importance_list.append(model.coef_.ravel())

    val_preds = model.predict_proba(x_val[select_feature])[:, 1]
    lr_auc += roc_auc_score(y_val, val_preds) / 5
    y_preds = model.predict(x_val[select_feature])
    lr_acc += accuracy_score(y_val, y_preds) / 5
    lr_test += model.predict_proba(x_test[select_feature])[:, 1] / 5
    lr_oof_1[val_idx] = val_preds
    lr_oof_2[val_idx] = y_preds

print(f"{Fore.GREEN}{Style.BRIGHT}Average auc = {round(lr_auc, 5)}, Average acc = {round(lr_acc, 5)}{Style.RESET_ALL}")
print(f"{Fore.RED}{Style.BRIGHT}OOF auc = {round(roc_auc_score(y, lr_oof_1), 5)}, OOF acc = {round(accuracy_score(y, lr_oof_2), 5)}{Style.RESET_ALL}")

submission['lr1'] = lr_test

select_feature = ['m3_missing', 'm5_missing', 'measurement_2', 'loading', 'measurement_17']

lr_oof_1 = np.zeros(len(train))
lr_oof_2 = np.zeros(len(train))
lr_test = np.zeros(len(test))
lr_auc = 0
lr_acc = 0
importance_list = []

#kf = GroupKFold(n_splits=5)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print("Fold:", fold_idx+1)
    x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    x_test = test.copy()
    
    x_train, x_val, x_test = _scale(x_train, x_val, x_test, select_feature)
    #x_train, x_val, x_test = x_train.round(2), x_val.round(2), x_test.round(2)
    
    model = LogisticRegression(max_iter=30000, C=0.00001, penalty='l2', solver='newton-cg') # , class_weight='balanced'
    model.fit(x_train[select_feature], y_train)
    importance_list.append(model.coef_.ravel())

    val_preds = model.predict_proba(x_val[select_feature])[:, 1]
    lr_auc += roc_auc_score(y_val, val_preds) / 5
    y_preds = model.predict(x_val[select_feature])
    lr_acc += accuracy_score(y_val, y_preds) / 5
    lr_test += model.predict_proba(x_test[select_feature])[:, 1] / 5
    lr_oof_1[val_idx] = val_preds
    lr_oof_2[val_idx] = y_preds

print(f"{Fore.GREEN}{Style.BRIGHT}Average auc = {round(lr_auc, 5)}, Average acc = {round(lr_acc, 5)}{Style.RESET_ALL}")
print(f"{Fore.RED}{Style.BRIGHT}OOF auc = {round(roc_auc_score(y, lr_oof_1), 5)}, OOF acc = {round(accuracy_score(y, lr_oof_2), 5)}{Style.RESET_ALL}")

submission['lr2'] = lr_test

submission.head()

select_feature = ['measurement_2', 'loading', 'measurement_17']

lr_oof_1 = np.zeros(len(train))
lr_oof_2 = np.zeros(len(train))
lr_test = np.zeros(len(test))
lr_auc = 0
lr_acc = 0
importance_list = []

#kf = GroupKFold(n_splits=5)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print("Fold:", fold_idx+1)
    x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    x_test = test.copy()
    
    x_train, x_val, x_test = _scale(x_train, x_val, x_test, select_feature)
    #x_train, x_val, x_test = x_train.round(2), x_val.round(2), x_test.round(2)
    
    model = LogisticRegression(max_iter=30000, C=0.00001, penalty='l2', solver='newton-cg') # , class_weight='balanced'
    model.fit(x_train[select_feature], y_train)
    importance_list.append(model.coef_.ravel())

    val_preds = model.predict_proba(x_val[select_feature])[:, 1]
    lr_auc += roc_auc_score(y_val, val_preds) / 5
    y_preds = model.predict(x_val[select_feature])
    lr_acc += accuracy_score(y_val, y_preds) / 5
    lr_test += model.predict_proba(x_test[select_feature])[:, 1] / 5
    lr_oof_1[val_idx] = val_preds
    lr_oof_2[val_idx] = y_preds

print(f"{Fore.GREEN}{Style.BRIGHT}Average auc = {round(lr_auc, 5)}, Average acc = {round(lr_acc, 5)}{Style.RESET_ALL}")
print(f"{Fore.RED}{Style.BRIGHT}OOF auc = {round(roc_auc_score(y, lr_oof_1), 5)}, OOF acc = {round(accuracy_score(y, lr_oof_2), 5)}{Style.RESET_ALL}")

submission['lr3'] = lr_test
submission[['id', 'lr0', 'lr1', 'lr2', 'lr3']].to_csv('best_model.csv', index=False)
