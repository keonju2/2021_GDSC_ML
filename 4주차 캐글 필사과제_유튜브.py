#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 분석에 필요한 패키지
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# plt의 스타일 지정
plt.style.use('seaborn')
sns.set(font_scale=2.5) 

# 결측치를 알기 쉽게 하는 패키지
import missingno as msno

# warning무시
import warnings
warnings.filterwarnings('ignore')

# notebook에서 바로 그림 확인하는 코드
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# 데이터 불러오기
df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')


# In[3]:


df_test.head()


# In[4]:


# 데이터 확인
df_train.head()


# In[5]:


# 데이터 행렬 확인
df_train.shape


# In[6]:


# 데이터 셋 특징 확인
df_train.describe()


# In[7]:


# max 값만 확인
df_train.max()


# In[8]:


df_test.describe()


# In[9]:


# column값 확인
df_train.columns


# In[10]:


# 각 column의 null 데이터 비율 확인 {:>10}:오른쪽 정렬
for col in df_train.columns:
    msg='column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col,100*(df_train[col].isnull().sum()/df_train[col].shape[0]))
    print(msg)


# In[11]:


# 값 확인
df_train[col]


# In[12]:


# null 값 확인
df_train[col].isnull()


# In[13]:


# null 값의 합
df_train[col].isnull().sum()


# In[14]:


# shape를 통해 총 데이터 갯수 확인하기
df_train[col].isnull().sum()/df_train[col].shape[0]


# In[15]:


for col in df_test.columns:
    msg='column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col,100*(df_test[col].isnull().sum()/df_test[col].shape[0]))
    print(msg)


# In[16]:


# missingno를 통해 확인하기
msno.matrix(df=df_train.iloc[:,:],figsize=(8,8),color=(0.8,0.5,0.2))


# In[17]:


# iloc으로 가져오고 싶은 위치 찾기
df_train.iloc[:,-1]


# In[18]:


msno.bar(df=df_train.iloc[:,:],figsize=(8,8),color=(0.8,0.5,0.2))


# In[19]:


# pie plot과 count-plot 그래프 그리기
# 도화지를 준비하는 과정 (1,2): 행렬 
f, ax=plt.subplots(1,2,figsize=(18,8)) 
#'Survived'에 있는 값 count하기, 떨어뜨리기, 글자 규칙, 그리는 위치, 그림자
df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Pie plot-Survived') # 제목
ax[0].set_ylabel('')
sns.countplot('Survived',data=df_train,ax=ax[1]) #countplot을 [1] 위치에 그리기
ax[1].set_title('Count plot-Survived')
plt.show()


# # 2.1 PClass

# In[20]:


# class 별 생존자 수 count는 객체가 몇명인가
df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).count()


# In[21]:


# sum은 숫자 자체의 데이터의 갯수 [0,1]에서 1을 다 더한 값
df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).sum()


# In[22]:


# crosstab을 통해 비교 (margin은 All 표현,style.background_gradient를 통해 색상 조절)
pd.crosstab(df_train['Pclass'],df_train['Survived'],margins=True).style.background_gradient(cmap='summer_r')


# In[23]:


# 평균 알아보기 (as_index를 통해 그래프 그리기 설정, sort_values를 통한 오름차순, ascending=False는 내림차순)
df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).mean().sort_values(by='Survived',ascending=False)


# In[24]:


# 그래프 기리기
df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).mean().sort_values(by='Survived',ascending=False).plot()


# In[25]:


# as_index=False일때는 Pclass도 같이 그린다
df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False).plot()


# In[26]:


# 막대그래프 그리기
df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar()


# In[27]:


y_position=1.02
f,ax=plt.subplots(1,2,figsize=(18,8))
# Class별 탑승자 수
df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number of passenger By Pclass',y=y_position)
ax[0].set_ylabel('Count')
# Class별 Survived와 Dead 구분 (hue를 통해 색깔 구분)
sns.countplot('Pclass',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead',y=y_position)
plt.show()


# # 2.2 Sex

# In[28]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df_train[['Sex','Survived']].groupby(['Sex'],as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()


# In[29]:


df_train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean()


# In[30]:


pd.crosstab(df_train['Sex'],df_train['Survived'],margins=True).style.background_gradient(cmap='summer_r')


# # 2.2 Both Sex and Pclass

# In[31]:


# factorplot 그래프 그리기
# 선은 error bar
sns.factorplot('Pclass','Survived',hue='Sex',data=df_train,size=6,aspect=1.5)


# In[32]:


# 축과 보는 방향을 바꾼 것
sns.factorplot(x='Sex',y='Survived',col='Pclass',data=df_train,saturation=.5,size=9,aspect=1)


# In[33]:


sns.factorplot(x='Sex',y='Survived',hue='Pclass',data=df_train,saturation=.5,size=9,aspect=1)


# # Age

# In[34]:


print('제일 나이 많은 탑승객: {:.1f} years'.format(df_train['Age'].max()))
print('제일 나이 어린 탑승객: {:.1f} years'.format(df_train['Age'].min()))
print('탑승객 평균 나이: {:.1f} years'.format(df_train['Age'].mean()))


# In[35]:


# kdeplot(커널 밀도 함수) 그리기 (히스토그램과 유사)
fig,ax=plt.subplots(1,1,figsize=(9,5))
sns.kdeplot(df_train[df_train['Survived']==1]['Age'],ax=ax)
sns.kdeplot(df_train[df_train['Survived']==0]['Age'],ax=ax)
plt.legend(['Survived'==1,'Survived'==0])
plt.show()


# In[36]:


# 히스토그램
df_train[df_train['Survived']==1]['Age'].hist()


# ## 그래프 그리는 다양한 방법

# In[37]:


f=plt.figure(figsize=(10,10))
a=np.arange(100)
b=np.sin(a)
plt.plot(b)


# In[38]:


f,ax=plt.subplots(1,1,figsize=(10,10))
a=np.arange(100)
b=np.sin(a)
plt.plot(b)


# In[39]:


plt.figure(figsize=(10,10))
a=np.arange(100)
b=np.sin(a)
plt.plot(b)


# In[40]:


# 탑승객의 연령별 분포
plt.figure(figsize=(8,6))
df_train['Age'][df_train['Pclass']==1].plot(kind='kde')
df_train['Age'][df_train['Pclass']==2].plot(kind='kde')
df_train['Age'][df_train['Pclass']==3].plot(kind='kde')
plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class','2nd Class','3rd Class'])


# In[41]:


# 히스토그램은 겹치면 보이지 않음
plt.figure(figsize=(8,6))
df_train['Age'][df_train['Pclass']==1].plot(kind='hist')
df_train['Age'][df_train['Pclass']==2].plot(kind='hist')
df_train['Age'][df_train['Pclass']==3].plot(kind='hist')
plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class','2nd Class','3rd Class'])


# In[42]:


fig,ax=plt.subplots(1,1,figsize=(9,5))
sns.kdeplot(df_train[(df_train['Survived']==0)&(df_train['Pclass']==1)]['Age'],ax=ax)
sns.kdeplot(df_train[(df_train['Survived']==1)&(df_train['Pclass']==1)]['Age'],ax=ax)
plt.legend(['Survived==1','Survived==0'])
plt.title('1st class')
plt.show()


# In[43]:


# 히스토그램은 겹치면 보이지 않음
plt.figure(figsize=(8,6))
df_train['Age'][(df_train['Pclass']==1)&(df_train['Survived']==0)].plot(kind='hist')
df_train['Age'][(df_train['Pclass']==1)&(df_train['Survived']==1)].plot(kind='hist')
plt.xlabel('Age')
plt.title('Age Distribution within classes')


# In[44]:


fig,ax=plt.subplots(1,1,figsize=(9,5))
sns.kdeplot(df_train[(df_train['Survived']==0)&(df_train['Pclass']==2)]['Age'],ax=ax)
sns.kdeplot(df_train[(df_train['Survived']==1)&(df_train['Pclass']==2)]['Age'],ax=ax)
plt.legend(['Survived==1','Survived==0'])
plt.title('2nd class')
plt.show()


# In[45]:


# 히스토그램은 겹치면 보이지 않음
plt.figure(figsize=(8,6))
df_train['Age'][(df_train['Pclass']==2)&(df_train['Survived']==0)].plot(kind='hist')
df_train['Age'][(df_train['Pclass']==2)&(df_train['Survived']==1)].plot(kind='hist')
plt.xlabel('Age')
plt.title('Age Distribution within classes')


# In[46]:


fig,ax=plt.subplots(1,1,figsize=(9,5))
sns.kdeplot(df_train[(df_train['Survived']==0)&(df_train['Pclass']==3)]['Age'],ax=ax)
sns.kdeplot(df_train[(df_train['Survived']==1)&(df_train['Pclass']==3)]['Age'],ax=ax)
plt.legend(['Survived==1','Survived==0'])
plt.title('3rd class')
plt.show()


# In[47]:


# 히스토그램은 겹치면 보이지 않음
plt.figure(figsize=(8,6))
df_train['Age'][(df_train['Pclass']==3)&(df_train['Survived']==0)].plot(kind='hist')
df_train['Age'][(df_train['Pclass']==3)&(df_train['Survived']==1)].plot(kind='hist')
plt.xlabel('Age')
plt.title('Age Distribution within classes')


# In[48]:


change_age_range_survival_ratio=[]

for i in range(1,80):
    change_age_range_survival_ratio.append(df_train[df_train['Age']<i]['Survived'].sum()/len(df_train[df_train['Age']<i]['Survived']))
    
plt.figure(figsize=(7,7))
plt.plot(change_age_range_survival_ratio)
plt.title('Survial rate change depending on range of Age',y=1.02)
plt.ylabel=('Survival rate')
plt.xlabel('Range of Age(0~x)')
plt.show()


# In[49]:


i=10
df_train[df_train['Age']<i]['Survived'].sum() / len(df_train[df_train['Age']<i]['Survived'])


# # Pclass, Sex, Age

# In[50]:


f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot('Pclass','Age',hue='Survived',data=df_train,scale='count',split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

sns.violinplot('Sex','Age',hue='Survived',data=df_train,scale='count',split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


# In[51]:


# split=False
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot('Pclass','Age',hue='Survived',data=df_train,scale='count',split=False,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

sns.violinplot('Sex','Age',hue='Survived',data=df_train,scale='count',split=False,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


# In[52]:


# scale 차이 같은 면적이기 때문에 count보다 숫자의 개념이 보기 힘듬
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot('Pclass','Age',hue='Survived',data=df_train,scale='area',split=False,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

sns.violinplot('Sex','Age',hue='Survived',data=df_train,scale='area',split=False,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


# # Embarked

# In[53]:


# Embarked 비율
f, ax= plt.subplots(1,1, figsize=(7,7))
df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar(ax=ax)


# In[54]:


# sort_values
df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=True).mean().sort_values(by='Survived')


# In[55]:


# 내림차순
df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=True).mean().sort_values(by='Survived',ascending=False)


# In[56]:


# sort_index
df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=True).mean().sort_index()


# In[57]:


f, ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked',data=df_train,ax=ax[0,0])
ax[0,0].set_title('(1) No. Of Passengers Boarded')

sns.countplot('Embarked', hue='Sex',data=df_train,ax=ax[0,1])
ax[0,1].set_title('(2) Male-Feamle split for embarked')

sns.countplot('Embarked', hue='Survived',data=df_train,ax=ax[1,0])
ax[1,0].set_title('(3) Embarked vs Survived')

sns.countplot('Embarked', hue='Pclass',data=df_train,ax=ax[1,1])
ax[1,1].set_title('(4) Embarked vs Pclass')

# 좌우간격, 상하간격 맞추기
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


# # Family - Sibsp + Parch

# In[58]:


df_train['FamilySize']=df_train['SibSp']+df_train['Parch']+1


# In[59]:


print('Maximum size of Family:',df_train['FamilySize'].max())
print('Minimum size of Family:',df_train['FamilySize'].min())


# In[60]:


f, ax=plt.subplots(1,3,figsize=(40,10))
sns.countplot('FamilySize',data=df_train,ax=ax[0])
ax[0].set_title('(1) No. Of Passenger Boarded',y=1.02)

sns.countplot('FamilySize',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilSize',y=1.02)

df_train[['FamilySize','Survived']].groupby(['FamilySize'],as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) Survived rate depending on FamilySize',y=1.02)

plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


# # Fare

# In[61]:


df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()
df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)


# In[62]:


fig, ax=plt.subplots(1,1,figsize=(8,8))
g=sns.distplot(df_train['Fare'],color='b',label='Skweness {:.2f}'.format(df_train['Fare'].skew()),ax=ax)
g=g.legend(loc='best')


# In[63]:


df_train['Fare']=df_train['Fare'].map(lambda i:np.log(i) if i>0 else 0)


# In[64]:


df_train['Ticket'].value_counts()


# # Fill Null in Age

# In[65]:


df_train['Age'].isnull().sum()


# In[66]:


df_train['Age'].mean()


# In[67]:


# str로 변환한 뒤 extract와 정규표현식을 통해 추출
df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.')
df_test['Initial']= df_test.Name.str.extract('([A-Za-z]+)\.') 


# In[68]:


pd.crosstab(df_train['Initial'],df_train['Sex']).T.style.background_gradient(cmap='summer_r')


# In[69]:


df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)


# In[70]:


df_train.groupby('Initial').mean()


# In[71]:


df_train.groupby('Initial')['Survived'].mean().plot.bar()


# In[72]:


df_train.loc[(df_train['Age'].isnull())&(df_train['Initial']=='Mr'),'Age']=33
df_train.loc[(df_train['Age'].isnull())&(df_train['Initial']=='Mrs'),'Age']=36
df_train.loc[(df_train['Age'].isnull())&(df_train['Initial']=='Master'),'Age']=5
df_train.loc[(df_train['Age'].isnull())&(df_train['Initial']=='Miss'),'Age']=22
df_train.loc[(df_train['Age'].isnull())&(df_train['Initial']=='Other'),'Age']=46

df_test.loc[(df_test['Age'].isnull())&(df_test['Initial']=='Mr'),'Age']=33
df_test.loc[(df_test['Age'].isnull())&(df_test['Initial']=='Mrs'),'Age']=36
df_test.loc[(df_test['Age'].isnull())&(df_test['Initial']=='Master'),'Age']=5
df_test.loc[(df_test['Age'].isnull())&(df_test['Initial']=='Miss'),'Age']=22
df_test.loc[(df_test['Age'].isnull())&(df_test['Initial']=='Other'),'Age']=46


# In[73]:


df_train.loc[(df_train['Initial']=='Mr'),'Age'].isnull().sum


# # Fill Null in Embarked and categorize Age

# In[74]:


df_train['Embarked'].isnull().sum()


# In[75]:


df_train['Embarked'].fillna('S',inplace=True)


# In[76]:


df_train['Embarked'].isnull().sum()


# In[77]:


df_train.head()


# In[78]:


df_train.loc[df_train['Age']<10,'Age_cat']=0
df_train.loc[(df_train['Age']>=10)&(df_train['Age']<20),'Age_cat']=1
df_train.loc[(df_train['Age']>=20)&(df_train['Age']<30),'Age_cat']=2
df_train.loc[(df_train['Age']>=30)&(df_train['Age']<40),'Age_cat']=3
df_train.loc[(df_train['Age']>=40)&(df_train['Age']<50),'Age_cat']=4
df_train.loc[(df_train['Age']>=50)&(df_train['Age']<60),'Age_cat']=5
df_train.loc[(df_train['Age']>=60)&(df_train['Age']<70),'Age_cat']=6
df_train.loc[df_train['Age']>=70,'Age_cat']=7


df_test.loc[df_test['Age']<10,'Age_cat']=0
df_test.loc[(df_test['Age']>=10)&(df_test['Age']<20),'Age_cat']=1
df_test.loc[(df_test['Age']>=20)&(df_test['Age']<30),'Age_cat']=2
df_test.loc[(df_test['Age']>=30)&(df_test['Age']<40),'Age_cat']=3
df_test.loc[(df_test['Age']>=40)&(df_test['Age']<50),'Age_cat']=4
df_test.loc[(df_test['Age']>=50)&(df_test['Age']<60),'Age_cat']=5
df_test.loc[(df_test['Age']>=60)&(df_test['Age']<70),'Age_cat']=6
df_test.loc[df_test['Age']>=70,'Age_cat']=7


# In[79]:


df_train.head()


# In[80]:


df_test.head()


# In[81]:


def category_age(x):
    if x<10:
        return 0
    elif x<20:
        return 1
    elif x<30:
        return 2
    elif x<40:
        return 3
    elif x<50:
        return 4
    elif x<60:
        return 5
    elif x<70:
        return 6
    else:
        return 7


# In[82]:


df_train['Age_cat_2']=df_train['Age'].apply(category_age)


# In[83]:


df_train.head()


# In[84]:


(df_train['Age_cat']==df_train['Age_cat_2']).all()


# In[85]:


df_train.drop(['Age','Age_cat_2'],axis=1,inplace=True)
df_test.drop(['Age'],axis=1,inplace=True)


# # Change string to categorical and Pearson coefficient

# In[86]:


df_train.Initial.unique()


# In[87]:


df_train.loc[df_train['Initial']=='Master','Initial']


# In[88]:


df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})


# In[89]:


df_train.Embarked.unique()


# In[90]:


df_train['Embarked'].value_counts()


# In[91]:


df_train['Embarked']=df_train['Embarked'].map({'C':0,'Q':1,'S':2})
df_test['Embarked']=df_test['Embarked'].map({'C':0,'Q':1,'S':2})


# In[92]:


df_train.head()


# In[93]:


df_train.Embarked.isnull().any()


# In[94]:


df_train['Sex'].unique()


# In[95]:


df_train['Sex']=df_train['Sex'].map({'female':0,'male':1})
df_test['Sex']=df_test['Sex'].map({'female':0,'male':1})


# In[96]:


heatmap_data=df_train[['Survived','Pclass','Sex','Fare','Embarked','FamilySize','Initial','Age_cat']]


# In[97]:


heatmap_data.corr()


# In[98]:


colormap=plt.cm.BuGn
plt.figure(figsize=(12,10))
plt.title('Pearson Correlation of Features',y=1.05,size=15)
sns.heatmap(heatmap_data.astype(float).corr(),linewidths=0.1,vmax=2,square=True,cmap=colormap,linecolor='white',annot=True,annot_kws={'size':16},fmt='.2f')


# # One-hot encoding on the Initial and Embarked

# In[99]:


df_test.head()


# In[100]:


df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')
df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')


# In[101]:


df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')
df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')


# In[102]:


df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)


# In[103]:


df_test.head()


# In[104]:


df_train.head()


# # Machine learningl(Randomforest)

# In[105]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[106]:


df_train.head()


# In[107]:


df_test.head()


# In[108]:


X_train=df_train.drop('Survived',axis=1).values
target_label=df_train['Survived'].values
X_test=df_test.values


# In[109]:


X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)


# In[110]:


model = RandomForestClassifier()
model.fit(X_tr, y_tr)
prediction = model.predict(X_vld)


# In[111]:


print('총 {}명 중 {:.2f}% 정확도로 생존 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))


# # feature importance and prediction on test set

# In[112]:


model.feature_importances_


# In[113]:


from pandas import Series

feature_importance = model.feature_importances_
Series_feat_imp = Series(feature_importance, index=df_test.columns)


# In[ ]:


plt.figure(figsize=(8, 8))
Series_feat_imp.sort_values(ascending=True).plot.barh()
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()


# In[ ]:


submission = pd.read_csv('gender_submission.csv')


# In[ ]:


submission.head()


# In[ ]:


prediction = model.predict(X_test)
submission['Survived'] = prediction


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




