import numpy as np
from matplotlib import pyplot as plt
import math
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

data = np.loadtxt('linear_regression.txt', delimiter=',')
# separate predictor from target variable
X = np.c_[np.ones(data.shape[0]), data[:, 0]]
y = np.c_[data[:, 1]]


# First appraoch - Normal equation

def normalEquation(X, y):
    """
    Parameteres: input variables (Table) , Target vector
    Instructions: Complete the code to compute the closed form solution to linear regression and 	save the result in theta.
    Return: coefficinets 
    """
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return theta


# Iterative Approach - Gradient Descent 

'''
Following paramteres need to be set by you - you may need to run your code multiple times to find the best combination 
'''


def compute_gradient(x, y, theta):
    n = len(x)
    gradient_0 = 2.0 / n * sum([(theta[0] + theta[1] * x[i] - y[i]) for i in range(n)])
    gradient_1 = 2.0 / n * sum([(theta[0] + theta[1] * x[i] - y[i]) * x[i] for i in range(n)])
    return [gradient_0, gradient_1]


def compute_new_theta(theta, direction, step):
    return [theta_i + step * direction_i for theta_i, direction_i in zip(theta, direction)]


def compute_distance(old, new):
    temp = [i - j for i, j in zip(old, new)]
    temp_sum = sum(i ** 2 for i in temp)
    return math.sqrt(temp_sum)


def compute_cost(x, y, theta):
    cost = 0.5 * sum([(j - (theta[0] + theta[1] * i)) ** 2 for i, j in zip(x, y)]) / len(x)
    return cost


def gradient_descent(l, x, y, tolerance=0.0000001, max_iter=100000):
    """
    Paramters: input variable , Target variable, theta, number of iteration, learning_rate
    Instructions: Complete the code to compute the iterative solution to linear regression, in each iteration you will 
    add the cost of the iteration to a an empty list name cost_hisotry and update the theta.
    Return: theta, cost_history 
    """

    # Your code goes here

    iter = 0
    theta = [0, 0]
    cost_history = []
    while True:
        gradient = compute_gradient(x, y, theta)
        next_theta = compute_new_theta(theta, gradient, l)

        if compute_distance(next_theta, theta) < tolerance:  # stop if we're converging
            break
        theta = next_theta  # continue if we're not
        cost = compute_cost(x, y, theta)
        cost_history.append(cost)
        iter += 1  # update iter

        if iter == max_iter:
            print('Max iteractions exceeded!')
            break

    return theta, cost_history


theta, cost_history = gradient_descent(-0.005, data[:, 0], data[:, 1])

# Plot the cost over number of iterations
'''
Your plot should be similar to the provided plot
'''

plt.plot(cost_history[0:1000])
plt.show()

# Plot the linear regression line for both gradient approach and normal equation in same plot
'''
hints: your x-axis will be your predictor variable and y-axis will be your target variable. plot a
scatter plot and draw the regression line using the theta calculated from both approaches. Your plot
should be similar to what provided.
'''

# Plot contour plot and 3d plot for the gradient descent approach

'''
your plots should be similar to our plots.

'''

theta_0 = np.arange(-10, 10, 0.25)
theta_1 = np.arange(-1, 4, 0.25)
theta_0, theta_1 = np.meshgrid(theta_0, theta_1)
Z = np.array(0.5 * sum([(j - (theta_0 + theta_1 * i)) ** 2 for i, j in data]) / len(data)).reshape(theta_0.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(theta_0, theta_1, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()

plt.contourf(theta_0, theta_1, Z, 90, alpha=0.5, cmap=cm.coolwarm)
plt.plot(theta[0], theta[1], 'ro', label="point")
plt.show()
