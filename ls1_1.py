import numpy as np
arr=np.random.randint(1,51,size=(5,4))
#part1
print(arr)
def antidia(a):
    b=[]
    for i in range(0,4):
        for k in range(0,4):
            if(i+k==3):
                b.append(int(a[i][k]))
    return b
print(antidia(arr))

#part2
for i in range(0,5):
    print(max(arr[i]))

#part3
avg=np.mean(arr)
arr1=[]
for i in arr:
    for j in i:
        if (j<=avg):
            arr1.append(int(j))
print(arr1)

#part4
def numpy_boundary_traversal(matrix):
    a=[]
    for i in range(0,4):
        a.append(int(arr[0][i]))
    for i in range(1,5):
        a.append(int(arr[i][3]))
    for i in range (2,-1,-1):
        a.append(int(arr[4][i]))
    for i in range(3,0,-1):
        a.append(int(arr[i][0]))
    return a
print(numpy_boundary_traversal(arr))