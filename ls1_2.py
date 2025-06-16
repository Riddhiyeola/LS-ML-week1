import numpy as np
arr=np.random.uniform(0,10,size=(20))

#part1
print(arr)
arr=np.round(arr,2)
print(arr)

#part2
print(np.max(arr))
print(np.min(arr))
print(np.median(arr))

#part3
a=[]
for i in range(0,20):
    if(arr[i]<5):
        a.append(float(arr[i]*arr[i]))
    else :
        a.append(float(arr[i]))
print(a)
a=np.round(a,2)
print(a)

#part4
def numpy_alternate_sort(array):
    b=[]
    c=0
    array1=array.copy()
    while(c<10):
        b.append(float(np.max(array1)))
        array1=np.delete(array1,np.argmax(array1),axis=None)
        b.append(float(np.min(array1)))
        array1=np.delete(array1,np.argmin(array1),axis=None)
        c+=1
    return b
print(numpy_alternate_sort(arr))
