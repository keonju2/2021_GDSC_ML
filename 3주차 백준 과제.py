#!/usr/bin/env python
# coding: utf-8

# # GDSC ML파트 3주차 과제로 백준 10문제를 풀어보았다.  

# ###### 중복되는 문제도 있습니다
# 
# ## 1157번 단어공부  
# 
# <https://www.acmicpc.net/problem/1157>  
# upper을 이용한 대문자 받기  
# count로 dictionary형태로 word 개수 세기
# 개수가 중복되는 단어들이 있으므로 max_list에서 따로 추출하기
# 조건에 맞게 최댓값이 하나면 알파벳을, 아니면 물음표를 출력하기

# In[74]:


word=input().upper()
count={}
for i in word:
    if i not in count:
        count[i]=0
    count[i]+=1
max_list=[j for j,k in count.items() if max(count.values())==k]
if len(max_list)==1:
    print(max_list[0])
else:
    print('?')


# ## 1546번 평균  
# 
# <https://www.acmicpc.net/problem/1546>  
# 
# map 함수를 통해서 점수 입력받기  
# new_mean()이라는 함수는 문제에서 나온 $점수/최대점수*100$ 이다.  
# map함수를 통해 new_mean함수를 score에 모두 적용해주고 sum을 통해 총합을 구한 뒤,
# count로 나눠주면 된다.

# In[19]:


count=int(input())
score=list(map(int,input().split())) # 점수 입력받기


# In[23]:


max_score=max(score) # 점수 최댓값 찾기
def new_mean(x): #최댓값과의 비율로 새로운 점수 만드는 함수
    return x/max_score*100


# In[29]:


print(sum(map(new_mean,score))/count) #평균을 구하는 식


# ## 2577번 숫자의 개수  
# 
# <https://www.acmicpc.net/problem/2577>  
# 
# 입력받는 숫자가 3개로 한정되어있으니까 for문을 통해서 세 숫자의 곱을 구했다.  
# count함수의 인덱스 0-9까지를 0-9숫자가 나왔을 때 하나씩 늘려주는 방법을 택했다.

# In[34]:


mul=1
for i in range(3): # 세 숫자의 곱 구하기
    a=int(input())
    mul=mul*a
print(mul)


# In[40]:


count=[0]*10 #0-9의 개수가 들어갈 list
for j in str(mul): #str(mul)로 해줘야 for문이 성립된다.
    for k in range(10): #0-9까지의 숫자를 확인하는 for문
        if int(j)==k:
            count[k]=count[k]+1 #숫자가 등장했을때 1 늘려주기
for l in count:
    print(l)


# ## 2675번 문자열 반복  
# 
# <https://www.acmicpc.net/problem/2675>  
# 
# 각 글자마다 R번 반복해서 출력해주는 문제이다.  
# case를 통해서 몇 번 반복할지를 결정해준다.
# word_list에 각 단어마다 R번씩 반복하여 append해준다.
# join으로 리스트에 있는 단어들을 문장으로 만들어준다.

# In[47]:


case=int(input()) # 시도할 횟수
for i in range(case): 
    R,P=input().split() # R,P입력받기
    word_list=[] #반복한 뒤 append해줄 list
    for j in P: #P에 있는 글자 순서대로 반복해주기
        for k in range(int(R)): #R을 str로 입력받아서 int로 변경해줘야함
            word_list.append(j) #반복된 글자를 append
    print(''.join(word_list)) #list안에 글자들 붙여서 출력해주기


# ## 2908번 상수  
# 
# <https://www.acmicpc.net/problem/2908>  
# 
# list(A)로 숫자들을 리스트화 해준다.  
# [::-1]로 거꾸로 뒤집어준다. A로 다시 저장하기 싫으면 reverse() 함수를 사용하면 된다.  
# join을 통해 숫자로 바꿔주고 int 형태로 바꿔준 뒤, max를 통해 최댓값을 찾는다.

# In[57]:


A,B=input().split() 
# A,B 숫자 거꾸로 만들기
A=list(A)[::-1] 
B=list(B)[::-1]
# A,B 다시 숫자로 만든 뒤 대소비교
A=int(''.join(A))
B=int(''.join(B))
print(max(A,B))


# ## 1018번 체스판 다시 칠하기  
# 
# <https://www.acmicpc.net/problem/1018>  

# ###### N,M 크기를  받고 보드 만들기  
# 
# nXm형태의 위치를 파악하기 쉽게 리스트 형태로 받았다.  

# In[75]:


n, m=map(int,input().split())
if 8<=n<=50 and 8<=m<=50:
    board = [input() for i in range(n)]


# ###### 위치가 짝수일 때와 홀수일 때로 나눠서  W, B가 아닐 때마다 점수를 추가해준 다음 가장 최소가 되는 값만 찾아내면 된다.  
# 
# 따라서 n * m의 보드에서 가능한 경우의 수는 n-7 * m-7이다. ex)10 13을 입력받을 경우 18가지.  
# 
# 8 * 8로 잘라주기 위해서 k와 l을 (i,i+8), (j,j+8)로 한정짓는다.  
# 
# k+l이 홀수일 경우와 짝수일 경우, W로 시작할 경우와 B로 시작할 경우를 나눠서 모든 경우의 수를 반복문으로 확인해준다.  
# 
# 마지막으로 total_score에 들어있는 값들 중 최솟값을 구해준다.  

# In[76]:


total_score=[]
# 보드에서 경우의 수 나누어주기
for i in range(n-7):
    for j in range(m-7):
        count_w=0 #w가 아닐때
        count_b=0 #b가 아닐때
        #8*8 크기로 잘라주기
        for k in range(i,i+8):
             for l in range (j,j+8):
                #각 경우의 수마다 비교해서 점수 추가하기
                if (k+l)%2==0:
                    if board[k][l]!='W':                            
                        count_w=count_w+1
                    if board[k][l]!='B':
                        count_b=count_b+1
                else:
                    if board[k][l]!='W':
                        count_b=count_b+1
                    if board[k][l]!='B':                            
                        count_w=count_w+1
        # 점수들 한 list에 모아주기
        total_score.append(count_w)
        total_score.append(count_b)
print(min(total_score)) #최솟값 출력


# ## 1436번 영화감독 숌  
# 
# <https://www.acmicpc.net/problem/1436>  

# ###### 666이 적어도 3개이상 연속으로 들어가는 수를 만든다  
# 
# 처음에 문제를 풀 때 중간에 666이 3개 이상 들어가는 경우를 제외해서 틀렸다.  
# 
# list666에 가장 작은 숫자인 666부터 '666'이 문자열로 들어가있는 숫자들을 확인해서 추가하였다.  
# 
# 입력받은 숫자가 list666의 길이보다 크면 계속 추가해주었고 list666[num-1]을 통하여 값을 출력해준다.  

# In[77]:


num=int(input())
list666=[]
i=666
while len(list666)<num:
    if '666' in str(i):
        list666.append(i)
    i=i+1
print(list666[num-1]) 


# ## 1259번 팰린드롬수  
#   
# <https://www.acmicpc.net/problem/1259>  

# ###### 앞에서 읽어도 뒤에서 읽어도 같은 숫자 찾기
# 
# 0을 입력하면 반복문이 끝나게 while과 if를 이용하였다.  
# 
# 입력받은 숫자는 위치를 찾기 편하게 문자형으로 입력받았다.  
# 
# 입력받은 숫자의 길이/2 만큼의 반복문을 돌리면 반대쪽은 (숫자의 길이-i-1)로 대응된다.  
# 
# 한가지 숫자라도 값이 다르면 False 값을 가지고 'no'를 출력하면 'yes'를 출력하는 것을 만들 때보다 길이가 짧아질 수 있다.  

# In[78]:


while True:
    word=input() # 숫자를 무한으로 입력받기 위해 while문 사용
    quest=True # 한가지 입력값을 처리하고나서 True, False값을 True로 초기화
    if word=='0': # 0을 입력하면 반복문 종료
        break
    else:
        word_len=len(word)
        for i in range(int((word_len)/2)): #단어 길이의 반만 확인하면 반대쪽 숫자와 대응된다.
                if word[i]!=word[word_len-1-i]: #반대쪽 숫자와 대응하기 위해서 word_len-1-i 사용
                    quest=False # 하나의 경우라도 False가 나오면 반복문 종료
                    continue
        if quest==False: # False가 나오면 바로 'no' 출력
            print('no')
        else: print('yes')


# ## 7568번 덩치  
# 
# <https://www.acmicpc.net/problem/7568>  
# 처음에 문제를 풀 때 너무 복잡하게 생각해서 무게 따로, 키 따로 점수 매기고 sort해서 index로 출력하려고 했는데 예제 문제는 옳게 나오지만 다른 경우에서 틀렸었다.  
# 또 and 말고 & 로 써서 한 번 더 틀렸는데 이건 bitwise 연산자라서 답이 다르게 나왔다.  

# In[1]:


# r값 입력 받고 (무게, 키) 형태로 리스트 만들기
count=int(input())
human=[]
for i in range(count):
    weight,tall=input().split()
    human.append((int(weight),int(tall)))
print(human)


# In[49]:


for j in human:
    rank=1 # 등수는 1등 부터니까 1
    for k in human:
        if j[0]<k[0] and j[1]<k[1]: # 둘 다 k가 우세할 경우에만 rank에 1 추가->순위 하락
            rank+=1
    print(rank,end=' ')


# In[52]:


for m in human:
    rank=1
    for n in human:
        if m[0]<n[0] & m[1]<n[1]: # &는 비트 연산자기 때문에 결과가 다르게 나온다.
            rank+=1
    print(rank,end=' ')


# ## 10250번 ACM호텔  
# 
# <https://www.acmicpc.net/problem/10250>  
# 1호가 우선시 된다면 H명씩 순서대로 채운다고 생각하면 편하다.  
# 따라서 층수는 N/H의 나머지가 되고 호수는 N/H의 몫+1이 된다.  
# 하지만 N이 H의 배수일 때, 층수가 최고층이 되기때문에 H를 대신 입력해준다.  
# 또한 호수도 N/H의 몫과 같아지기 때문에 +1 한 것을 다시 -1 해준다.  
# 또한 호수가 10보다 작을 때 앞에 0을 적어줘야한다.

# In[73]:


for i in range(int(input())):
    H,W,N=map(int,input().split())
    floor=N%H #나머지가 층수가 된다.
    order=(N//H)+1 #몫을 정수로 받은 뒤 +1해주면 호수가 된다.
    if floor==0: #N이 H의 배수일 때만 따로 구분해서 값을 준다.
        floor=H
        order=order-1
    if order<10: # 호수 앞에 0을 붙여준다.
        print(str(floor)+'0'+str(order))
    else:
        print(str(floor)+str(order))


# In[ ]:





# In[ ]:





# In[ ]:




