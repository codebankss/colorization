import random
import numpy as np
from numpy import linalg as LA

class Network(object):

    def __init__(self, sizes, std):
        self.cnn_layers = len(sizes)
        self.sizes = sizes
        self.filter1 = [std*np.random.randn(i, 1) for i in sizes[1:]]
        self.filter2 = [1/np.sqrt(i)*np.random.randn(j, i)
                        for i, j in zip(sizes[:-1], sizes[1:])]
        self.highlight_1 = [1 for i in sizes[:-1]]
        self.highlight_2 = [std*np.zeros((i, 1)) for i in sizes[:-1]]

    def conv(self, i):
        for b, w, high1, high2 in zip(self.filter1, self.filter2, self.highlight_1, self.highlight_2):
            i, temp = image_batch(i, 1e-5, high1, high2)
            i = sigmoid(np.dot(w, i)+b)
        return i

    def CNN(self, training_data, epochs, image_subset_size, alpha,beta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            image_subsets = [training_data[k:k+image_subset_size]
                for k in range(0, n, image_subset_size)]
            for image_subset in image_subsets:
                self.applying_filter(image_subset, alpha,beta)
            if test_data:
                print ("Epoch {0}: {1} / {2}").format(i, self.evaluate(test_data), n_test)
            else:
                print("Epochs completed "+str(i+1))
                print(self.cost(training_data))
                print(self.highlight_1)

    def applying_filter(self, image_subset, alpha,beta):
        init_b = [np.zeros(b.shape) for b in self.filter1]
        init_w = [np.zeros(w.shape) for w in self.filter2]
        init_high1 = [0 for high1 in self.highlight_1]
        init_high2 = [np.zeros(high2.shape) for high2 in self.highlight_2]
        for i, j in image_subset:
            delta_init_b, delta_init_w, delta_nabla_high1, \
                                    delta_init_high2  = self.convolve(i, j)
            init_b = [nb+dnb for nb, dnb in zip(init_b, delta_init_b)]
            init_w = [nw+dnw for nw, dnw in zip(init_w, delta_init_w)]
            init_high1 = [n_high1+dn_high1 for n_high1, dn_high1 \
                               in zip(init_high1, delta_nabla_high1)]
            init_high2 = [n_high2+dn_high2 for n_high2, dn_high2 \
                                  in zip(init_high2, delta_init_high2)]
        self.filter2 = [w-(alpha/len(image_subset))*nw-alpha*beta*w\
                        for w, nw in zip(self.filter2, init_w)]
        self.filter1 = [b-(alpha/len(image_subset))*nb \
                       for b, nb in zip(self.filter1, init_b)]
        self.highlight_1 = [g-(alpha/len(image_subset))*ng-alpha*beta*g\
                       for g, ng in zip(self.highlight_1, init_high1)]
        self.highlight_2 = [b-(alpha/len(image_subset))*nb \
                      for b, nb in zip(self.highlight_2, init_high2)]

    def convolve(self, i, j):
        init_b = [np.zeros(b.shape) for b in self.filter1]
        init_w = [np.zeros(w.shape) for w in self.filter2]
        init_high1 = [0 for g in self.highlight_1]
        init_high2 = [np.zeros(b.shape) for b in self.highlight_2]
        k = 1e-5 
        
        activation = i
        activations = [i] 
        xarr = [] 
        color_dimension = [] 
        trained_images_parameter = [] 
        for b, w, high1, high2 in zip(self.filter1, self.filter2, \
                                     self.highlight_1, self.highlight_2):
            xb, param = image_batch(activation, k, high1, high2)
            trained_images_parameter.append(param)
            xarr.append(xb)
            z = np.dot(w, xb)+b
            color_dimension.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        derivative = self.cost_derivative(activations[-1], j) * \
        sigmoid_dash(color_dimension[-1])
        init_b[-1] = derivative
        init_w[-1] = np.dot(derivative, activations[-2].transpose())
        for l in range(2, self.cnn_layers):
            z = color_dimension[-l]
            dout = np.dot(self.filter2[-l+1].transpose(), derivative)
            delta_b, init_high1[-l+1], init_high2[-l+1] = image_batch_backward_pooling \
            (dout, trained_images_parameter[-l+1], self.highlight_1[-l+1], self.highlight_2[-l+1], k)
            sp = sigmoid_dash(z)
            derivative =  delta_b * sp
            init_b[-l] = derivative
            init_w[-l] = np.dot(derivative, activations[-l-1].transpose())
        dout = np.dot(self.filter2[0].transpose(), derivative)
        delta_b, init_high1[0], init_high2[0] = image_batch_backward_pooling(dout, \
                            trained_images_parameter[0],self.highlight_1[0],self.highlight_2[0],k)
        return (init_b, init_w, init_high1, init_high2)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.conv(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost(self, training_data):
        error = [self.conv(x)-y
                        for (x, y) in training_data]
        squared_error = 0
        for x in error:
            squared_error = squared_error + LA.norm(x)
        return squared_error

    def cost_derivative(self, output_activations, y):
        
        output_activations
        return (output_activations-y)

def image_batch(i,j,high1,high2):
    avg = np.mean(i)
    deriv = i - avg
    variance = np.sum((i - avg)**2)/(len(i)-1)
    root_variance = np.sqrt(variance+j)
    inverse = 1/root_variance
    x_bar = deriv*inverse
    out = high1*x_bar + high2
    param = (x_bar, avg, inverse, root_variance, variance)
    return out, param

def image_batch_backward_pooling(dout, param, high1, high2, eps):
  x_bar,x_avg,variance,root_variance,variance_ = param
  N,D = np.shape(dout)
  dhigh2 = dout
  dhigh1x = dout 
  dhigh1 = np.sum(dhigh1x*x_bar, axis=0)
  dx_bar = dhigh1x * high1
  dvariance = np.sum(dx_bar*x_avg, axis=0)
  dx_avg1 = dx_bar * variance
  droot_variance = -1. /(root_variance**2) * dvariance
  dvar = 0.5 * 1. /np.sqrt(variance_+eps) * droot_variance
  dsq = 1. /N * np.ones((N,D)) * dvar
  dx_avg2 = 2 * x_avg * dsq
  dx1 = (dx_avg1 + dx_avg2)
  dmu = -1 * np.sum(dx_avg1+dx_avg2, axis=0)
  dx2 = 1. /N * np.ones((N,D)) * dmu
  dx = dx1 + dx2
  return dx, dhigh1, dhigh2

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_dash(z):
    return sigmoid(z)*(1-sigmoid(z))
