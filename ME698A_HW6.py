import numpy as np

def function(x):
    return (1-x[1])**2+100*(x[2]-x[1]**2)**2  
def Z(w,x):
    z=np.zeros((3), dtype="float")
    z[0]=1
    z[1]=Relu(w[0][0]+w[0][1]*x[1]+w[0][2]*x[2])
    z[2]=Relu(w[1][0]+w[1][1]*x[1]+w[1][2]*x[2])
    return z
def y_hatf(z,h):
    return Relu(h[0]*z[0]+z[1]*h[1]+z[2]*h[2])
def Relu(k):
    if k>0:
        return k
    else:
        return 0

def derivative(a):
    if a>0:
        return 1
    else:
        return 0

train_x=np.random.rand(200,3)
for i in range(200):
    train_x[i][0]=1
train_y=np.zeros((200),dtype="float")
for i in range(200):
    train_y[i]=function(train_x[i])

w=np.random.rand(2,3)
h=np.random.rand(3)
r=0.00001
epsilon=0.1
w_old=w

y_hat_array=np.zeros(200,dtype="float")
epoch=100000
for epoch in range(epoch):
    Loss=0
    for i in range(200):
        L=1
        iteration=0
        z=Z(w_old,train_x[i])
        y_hat=y_hatf(z,h)
        h_new=h
        h_old=h
        d_h=np.zeros(3,dtype="float")
        d_w=np.zeros((2,3),dtype="float")
        while(L>epsilon and iteration<1):
            iteration+=1
            for j in range(3):
                d_h[j]=2*(train_y[i]-y_hat)*derivative(y_hat)*z[j]
            for k in range(2):
                for l in range(3):
                    d_w[k][l]=2*(train_y[i]-y_hat)*derivative(y_hat)*h_new[k+1]*derivative(z[k+1])*train_x[i][l]       
            w_new=w_old+r*d_w
            h_new=h_old+r*d_h
            z=Z(w_new,train_x[i])
            y_hat=y_hatf(z,h_new)
            L=(train_y[i]-y_hat)**2/200
            w_old=w_new
        y_hat_array[i]=y_hat
        Loss=Loss+L 
        w=w_old
        h=h_new

print("weights=", w )
print("hidden weights=", h)
print("average training loss=", Loss)

test_x=np.random.rand(100,3)
for i in range(100):
    test_x[i][0]=1
test_y=np.zeros((100),dtype="float")
for i in range(100):
    test_y[i]=function(test_x[i])

Loss=0
for i in range(100):
    z=Z(w,test_x[i])
    y_hat=y_hatf(z,h)
    Loss+=(test_y[i]-y_hat)**2
Loss=Loss/100
print("average test loss=", Loss)






