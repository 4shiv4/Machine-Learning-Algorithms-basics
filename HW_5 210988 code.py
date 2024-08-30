import numpy as np
import matplotlib.pyplot as plt

x_in1 = np.array([15.5,23.75,8,17,5.5,19,24,2.5,7.5,11])
x_in2 = np.array([40,23.25,17,21,10,12,20,12,15,26])

y_in = np.array([0,0,1,0,1,1,0,1,1,0],dtype=np.int8)

def sigmoid(z):
    return 1/(1 + np.exp(-z))
def objective_function(x):
    z1=np.log(1+np.exp(-x[0]-x[1]*x_in1-x[2]*x_in2))
    z2=np.log(1+np.exp(x[0]+x[1]*x_in1+x[2]*x_in2))
    return np.dot(y_in,z1)+np.dot((1-y_in),z2)
def gradient_function(x):
    z1=1/(1+np.exp(x[0]+x[1]*x_in1+x[2]*x_in2))
    z2=1/(1+np.exp(-x[0]-x[1]*x_in1-x[2]*x_in2))
    return np.array([-np.dot(y_in, z1) + np.dot((1 - y_in), z2), -np.dot(y_in * x_in1, z1) + np.dot((1 - y_in) * x_in1, z2), -np.dot(y_in * x_in2, z1) + np.dot((1 - y_in) * x_in2, z2)])
    
def line_search(objective_function,gradient,x):
    beta=.1
    stepsize=1
    trial=100
    tau=.5
    for i in range(trial):
        fx1=objective_function(x)
        fx2=objective_function(x-stepsize*gradient)
        c=-beta*stepsize*np.dot(gradient,gradient)
        if fx2-fx1 <=c:
            break
        else:
            stepsize=tau*stepsize
    return stepsize
    
maxit=10000;epsilon=1.e-6
x=np.array([-2,3,1])
for i in range(maxit):
    gradient=gradient_function(x);b=np.linalg.norm(gradient)
    if b < epsilon:
        break
    stepsize=line_search(objective_function,gradient,x)
    x = x -stepsize * gradient
#    print(i,b)

minimum_value = objective_function(x)

print("Minimum value:", minimum_value)
print("Minimum location:", x)
print("iteration:", i+1)

x1 = np.linspace(min(x_in1), max(x_in1), 100)
x2 = np.linspace(min(x_in2), max(x_in2), 100)
xx1, xx2 = np.meshgrid(x1, x2)
hypo_vals = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(x))
passing_probability = hypo_vals.reshape(xx1.shape)


plt.figure(figsize=(8, 6))
plt.scatter(x_in1[y_in==1], x_in2[y_in==1], color='blue', label='Pass')
plt.scatter(x_in1[y_in==0], x_in2[y_in==0], color='red', label='Fail')
contour = plt.contourf(xx1, xx2, passing_probability, levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.5)
plt.xlabel('Propellant Age (Weeks)')
plt.ylabel('Storage Temperature (°C)')
plt.colorbar(contour, label='Passing Probability')
plt.legend()
dpi = 300
figure = 'Passing_Probability.png'
plt.tight_layout()
plt.savefig(figure, dpi=dpi)
plt.show()