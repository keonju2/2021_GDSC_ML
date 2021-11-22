#!/usr/bin/env python
# coding: utf-8

# 1. Visual inspection of your data
# 2. Defining the metadata
# 3. Descriptive statistics
# 4. Handling imbalanced classes
# 5. Data quality checks
# 6. Exploratory data visualization
# 7. Feature engineering
# 8. Feature selection
# 9. Feature scaling

# In[1]:


## 패키지 설치

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 100)


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# ## 데이터 확인하기  
# 
# 유사한 그룹에 속하는 feature은 태그가 지정됩니다. (ind,reg,car,calc)  
# feature 이름에서 bin은 이진 특성을, cat은 범주형 특성입니다.  
# 그 외에는 연속 또는 순서형 특성입니다.  
# -1은 결측값입니다.  

# In[3]:


train.head()


# In[4]:


train.tail()


# In[5]:


train.shape


# In[6]:


train.drop_duplicates()
train.shape


# In[7]:


test.shape 


# In[8]:


train.info()


# float인지 int인지 자료형을 알 수 있으며 null 대신 -1이 들어갔으므로 null이 없습니다.  

# ## 메타 데이터
# 
# 메타데이터를 통해 데이터 정보를 저장합니다.  
# 변수의 타입과 feature의 특성을 저장합니다.  

# In[9]:


data = []
for f in train.columns:
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'
         
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif train[f].dtype == float:
        level = 'interval'
    elif train[f].dtype == int:
        level = 'ordinal'
        
    keep = True
    if f == 'id':
        keep = False
    
    dtype = train[f].dtype
    
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)
    
meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)


# In[10]:


meta


# In[11]:


meta[(meta.level == 'nominal') & (meta.keep)].index


# In[12]:


pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()


# ## 통계량 살펴보기
# 
# 메타 데이터에서 interval인 값만 찾아서 describe를 사용할 수 있습니다.  
# ps_reg_03, ps_car_12, ps_car_15에 결측값이 있습니다.  

# In[13]:


v = meta[(meta.level == 'interval') & (meta.keep)].index
train[v].describe()


# In[14]:


v = meta[(meta.level == 'binary') & (meta.keep)].index
train[v].describe()


# target=1인 비율이 target=0보다 훨씬 작기 때문에 target=1을 오버 샘플링 혹은 target=0으로 언더샘플링 할 수 있습니다.  
# 언더 샘플링은 불균형한 데이터 셋에서 높은 비율을 차지하던 클래스의 데이터 수를 줄임으로써 데이터 불균형을 해소하는 아이디어 입니다.  
# 하지만 이 방법은 학습에 사용되는 전체 데이터 수를 급격하게 감소시켜 오히려 성능이 떨어질 수 있습니다.  
# 오버 샘플링은 낮은 비율 클래스의 데이터 수를 늘림으로써 데이터 불균형을 해소하는 아이디어 입니다.  

# In[15]:


desired_apriori=0.10

idx_0 = train[train.target == 0].index
idx_1 = train[train.target == 1].index

nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])

undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
undersampled_nb_0 = int(undersampling_rate*nb_0)
print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))

undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

idx_list = list(undersampled_idx) + list(idx_1)

train = train.loc[idx_list].reset_index(drop=True)


# ps_car_03_cat 및 ps_car_05_cat는 결측값이 있는 비율이 높습니다. 이 변수를 제거합니다.  
# 결측값이 있는 다른 범주형 변수의 경우 결측값 -1을 그대로 둘 수 있습니다.  
# ps_reg_03 (continuous)에는 18%에 대한 결측값이 있습니다. 평균으로 대체합니다.  
# ps_car_11 (순서형)에는 결측 값이 5개만 있습니다. 모드로 대체합니다.  
# ps_car_12 (연속)에는 결측값이 1개만 있습니다. 평균으로 대체합니다.  
# ps_car_14 (연속)에는 7%에 대한 결측값이 있습니다. 평균으로 대체합니다.  

# In[16]:


vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train.drop(vars_to_drop, inplace=True, axis=1)
meta.loc[(vars_to_drop),'keep'] = False  # 메타 데이터 update

mean_imp = SimpleImputer(missing_values=-1, strategy='mean')
mode_imp = SimpleImputer(missing_values=-1, strategy='most_frequent')
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()


# In[17]:


v = meta[(meta.level == 'nominal') & (meta.keep)].index

for f in v:
    dist_values = train[f].value_counts().shape[0]
    print('Variable {} has {} distinct values'.format(f, dist_values))


# In[18]:


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # 평균 계산
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    
    prior = target.mean()

    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)

    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


# In[19]:


train_encoded, test_encoded = target_encode(train["ps_car_11_cat"], 
                             test["ps_car_11_cat"], 
                             target=train.target, 
                             min_samples_leaf=100,
                             smoothing=10,
                             noise_level=0.01)
    
train['ps_car_11_cat_te'] = train_encoded
train.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat','keep'] = False  # 메타 데이터 업데이트
test['ps_car_11_cat_te'] = test_encoded
test.drop('ps_car_11_cat', axis=1, inplace=True)


# ## 탐색 데이터 시각화

# 범주값으로 결측값을 유지한다면 비율을 확인하기 좋습니다.  

# In[21]:


v = meta[(meta.level == 'nominal') & (meta.keep)].index

for f in v:
    plt.figure()
    fig, ax = plt.subplots(figsize=(20,10))
    cat_perc = train[[f, 'target']].groupby([f],as_index=False).mean()
    cat_perc.sort_values(by='target', ascending=False, inplace=True)
    sns.barplot(ax=ax, x=f, y='target', data=cat_perc, order=cat_perc[f])
    plt.ylabel('% target', fontsize=18)
    plt.xlabel(f, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show();


# 구간 변수는 상관 관계를 확인하는 것이 좋습니다.  

# In[22]:


def corr_heatmap(v):
    correlations = train[v].corr()

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.show();
    
v = meta[(meta.level == 'interval') & (meta.keep)].index
corr_heatmap(v)


# In[23]:


s = train.sample(frac=0.1)


# In[24]:


sns.lmplot(x='ps_reg_02', y='ps_reg_03', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()


# In[25]:


sns.lmplot(x='ps_car_12', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()


# In[26]:


sns.lmplot(x='ps_car_12', y='ps_car_14', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()


# In[27]:


sns.lmplot(x='ps_car_15', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()


# ## Feature Engineering

# In[29]:


# 더미 변수 만들기
v = meta[(meta.level == 'nominal') & (meta.keep)].index
print('Before dummification we have {} variables in train'.format(train.shape[1]))
train = pd.get_dummies(train, columns=v, drop_first=True)
print('After dummification we have {} variables in train'.format(train.shape[1]))


# In[30]:


# 상호작용 변수 만들기
v = meta[(meta.level == 'interval') & (meta.keep)].index
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
interactions = pd.DataFrame(data=poly.fit_transform(train[v]), columns=poly.get_feature_names(v))
interactions.drop(v, axis=1, inplace=True)
print('Before creating interactions we have {} variables in train'.format(train.shape[1]))
train = pd.concat([train, interactions], axis=1)
print('After creating interactions we have {} variables in train'.format(train.shape[1]))


# ## Feature Selection

# In[31]:


# VarianceThreshold을 통해 분산이 작은 값 제거
selector = VarianceThreshold(threshold=.01)
selector.fit(train.drop(['id', 'target'], axis=1))
f = np.vectorize(lambda x : not x) 
v = train.drop(['id', 'target'], axis=1).columns[f(selector.get_support())]
print('{} variables have too low variance.'.format(len(v)))
print('These variables are {}'.format(list(v)))


# In[32]:


# 랜덤 포레스트를 통한 특성 선택 및 모델 선택

X_train = train.drop(['id', 'target'], axis=1)
y_train = train['target']

feat_labels = X_train.columns

rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

rf.fit(X_train, y_train)
importances = rf.feature_importances_

indices = np.argsort(rf.feature_importances_)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))


# In[33]:


sfm = SelectFromModel(rf, threshold='median', prefit=True)
print('Number of features before selection: {}'.format(X_train.shape[1]))
n_features = sfm.transform(X_train).shape[1]
print('Number of features after selection: {}'.format(n_features))
selected_vars = list(feat_labels[sfm.get_support()])


# In[34]:


# Feature scaling
scaler = StandardScaler()
scaler.fit_transform(train.drop(['target'], axis=1))

