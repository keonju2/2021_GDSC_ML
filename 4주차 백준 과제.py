#!/usr/bin/env python
# coding: utf-8

# # GDSC ML파트 4주차 과제로 백준 5문제를 풀어보았다.  

# ###### 중복되는 문제도 있습니다
# 
# ## 3085번 사탕 게임  
# 
# <https://www.acmicpc.net/problem/3085>  
# 먼저 보드는 한 글자당 리스트 한 요소를 차지하게끔 이중으로 만든다.  
# check 함수를 통해 보드에 연속된 사탕이 몇개인지 만들어준다.  
# 그 다음 for문을 통해 값들을 하나씩 변경해보고 가장 많은 값을 찾는다.  

# In[23]:


num=int(input())
board=[]
answer=0
for i in range(num):
    candy=list(input())
    board.append(candy)


# In[25]:


def check(board):
    n=len(board)
    answer=1 #연속된 사탕의 결과
    
    for i in range(n):
        count=1
        for j in range(1, n):
            if board[i][j] == board[i][j-1]:  #열에서 같다면 +1 해주기
                count += 1
            else:
                count=1    # 같지 않다면 1로 초기화
            if count > answer:
                answer = count # 가장 큰 값을 answer로 반환
        count=1 # 행 결과를 찾기 위한 초기화
        for j in range(1, n):
            if board[j][i] == board[j-1][i]: # 행에서 최대값 찾기
                count += 1
            else:
                count=1
            if count > answer:
                answer = count
    return answer


# In[27]:


answer=0
for i in range(num):
    for j in range(num):
        if j+1 < num:
            board[i][j],board[i][j+1] =board[i][j+1],board[i][j] # 열에서 값 바꾸기
            temp=check(board) # 최대 개수 확인하기

            if temp > answer:
                answer = temp # 가장 많은 값으로 저장
            board[i][j], board[i][j+1] = board[i][j+1], board[i][j] # 값 초기화하기


        if i+1 < num: #행에서 마찬가지로 진행
            board[i][j], board[i+1][j] = board[i+1][j], board[i][j]
            temp=check(board)

            if temp > answer:
                answer = temp
            
            board[i][j], board[i+1][j] = board[i+1][j], board[i][j]
            
print(answer)


# ## 2563번 색종이
# 
# <https://www.acmicpc.net/problem/2563>  
# 
# 100 * 100의 흰 색종이를 먼저 만들어준다.  
# 다음 입력받은 색종이만큼 0을 1로 바꿔주면 중복도 해결하면서 검은색을 표시할 수 있다.  
# 1이 된 숫자의 부분만 세면 된다.  

# In[35]:


paper=[[0 for i in range(101)] for j in range(101)]
for i in range(int(input())):
    x,y=map(int,input().split())
    for j in range(x,x+10):
        for k in range(y,y+10):
            paper[j][k]=1

result=0
for i in paper:
    result += i.count(1)
print(result)


# ## 4673번 색종이
# 
# <https://www.acmicpc.net/problem/4673>  
# 
# for문을 통해 숫자들의 셀프 넘버를 구해서 초기 [1:10000]의 리스트에서 셀프 넘버가 나오면 제거해주었다.

# In[42]:


self_num=[i for i in range(1,10001)]

for i in range(1,10001):
    total=i
    num=str(i)
    for j in range(len(num)):
        total=total+int(num[j])
    if total in self_num:
        self_num.remove(total)
for i in range(len(self_num)):
    print(self_num[i])


# ## 5635번 생일
# 
# <https://www.acmicpc.net/problem/5635>  
# 
# for문을 통해 글자들을 입력받고 연-월-일 부분을 정수형으로 치환한다.  
# sort와 lambda를 이용해서 오름차순으로 정렬하면 가장 마지막 사람이 어리고 가장 처음 사람이 나이가 가장 많을 것이다.  

# In[64]:


people=[]
for i in range(int(input())):
    person=list(input().split(' '))
    person[1]=int(person[1])
    person[2]=int(person[2])
    person[3]=int(person[3])
    people.append(person)

people.sort(key=lambda x:[x[3],x[2],x[1]])

print(people[-1][0])
print(people[0][0])


# ## 11170번 0의 개수
# 
# <https://www.acmicpc.net/problem/11170>  
# 
# a,b 두 숫자 사이의 모든 숫자들을 붙여서 하나의 글자로 만들어준다음 count함수를 사용하였다.  

# In[82]:


for i in range(int(input())):
    a,b=input().split()
    
    word=''
    for j in range(int(a),int(b)+1):
        word=word+str(j)
    
    print(word.count('0'))

