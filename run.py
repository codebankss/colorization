import numpy as np
import matplotlib.pyplot as plt
import os.path as p
from skimage import color
import matplotlib.image as mpimg
import pickle


training_samples = 1000 # Numer of images considering for training
size_of_image = 32 


# Reading image and resizing
def reading_data(training_samples, size_of_image):
    output = np.zeros((size_of_image*size_of_image*3,training_samples)) 
    input_ = np.zeros((size_of_image*size_of_image,training_samples)) # Flattening input image
    gray_image = np.zeros((size_of_image,size_of_image,training_samples)) # Array for gray image
    original_image = np.zeros((size_of_image,size_of_image,3,training_samples)) # Array for original image

    for k in range(1,training_samples+1):
        image_name = str(k) + '.png'
        if p.isfile('train/'+image_name):
            image = mpimg.imread('train/'+image_name)
            image_gray = color.rgb2gray(image)
            gray_image[:,:,k-1] = image_gray
            original_image[:,:,:,k-1] = image
            output[:,k-1] = np.reshape(image,(size_of_image*size_of_image*3,))
            input_[:,k-1] = np.reshape(image_gray,(size_of_image*size_of_image,))
        else:
            print('File not found in '+'train/'+image_name)
            break
    return (input_, output)

input_,output = reading_data(training_samples,size_of_image)




def compiling_data(input_,output,training_samples,size_of_image):
    training_inputs = [np.reshape(input_[:,i],(size_of_image*size_of_image,1)) for i in range(training_samples)]
    training_results = [np.reshape(output[:,i],(size_of_image*size_of_image*3,1)) for i in range(training_samples)]
    training_data = list(zip(training_inputs, training_results))
    return training_data

training_data = compiling_data(input_,output,training_samples,size_of_image)




# CNN
'''
import network
s = 1e-4;
cnn_obj = network.Network([size_of_image*size_of_image,1024,size_of_image*size_of_image*3],s)
print ('CNN!')
cnn_obj.CNN(training_data,500,10,3e-3,0)

#Dumping the cnn_obj object to pickle

with open('cnn.pkl', 'wb') as output:
    pickle.dump(cnn_obj, output, pickle.HIGHEST_PROTOCOL)

'''

pickle_in = open('cnn.pkl', 'rb')
cnn_obj = pickle.load(pickle_in)


# Coloring
print ('Initialized')
image_for_testing = np.reshape(input_[:,0],(size_of_image*size_of_image,1))
testing_output = cnn_obj.conv(image_for_testing)
testing_original_image = np.reshape(output[:,0],(size_of_image*size_of_image*3,1))
output_iimage = np.reshape(testing_output,(size_of_image,size_of_image,3))
gray_image = np.reshape(image_for_testing,(size_of_image,size_of_image))
original_image = np.reshape(testing_original_image,(size_of_image,size_of_image,3))

# Output plots
plt.figure()
plt.subplot(1,3,1)
plt.imshow(original_image,cmap='gray')
plt.title('Image')
plt.subplot(1,3,2)
plt.imshow(gray_image,cmap='gray')
plt.title('Gray image')
plt.subplot(1,3,3)
plt.imshow(output_iimage)
plt.title('Image colored by CNN')
plt.show()