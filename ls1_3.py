import pandas as pd
import numpy as np
name=[]
for i in range(10):
    name.append(input())
sub=np.random.choice(['maths','physics','chemistry','biology'],size=10)
score=np.random.randint(50,101,size=10)
grade=[]

for i in score:
    if(i<=100 and i>=90):
        grade.append('A')
    elif(i<90 and i>79):
        grade.append('B')
    elif(i<80 and i>69):
        grade.append('C')  
    elif(i<70 and i>59):
        grade.append('D')
    else:
        grade.append('F')
    
df=pd.DataFrame(
    {
        'Name':name,
        'Subject':sub,
        'Score': score,
        'Grade': grade
    }
)

print(df.sort_values(by='Score',ascending=False))
avg=df.groupby('Subject')['Score'].mean().round(2)
print(avg)

def pandas_filter_pass(dataframe):
    return dataframe[dataframe['Grade'].isin(['A','B'])]
print(pandas_filter_pass(df))
