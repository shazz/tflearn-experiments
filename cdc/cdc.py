#
# Lesson #1 challenge from Sirajology - Introduction - Learn Python for Data Science #1
# https://www.youtube.com/watch?v=T5pRlIbr6gg
# Added a few classifiers
# Using CDC BRFSS Annual Survey Data 2015 
# https://www.cdc.gov/brfss/annual_data/annual_2015.html

# (C) 2016 - Shazz 
# Under MIT license

# classifier
import tflearn
import data_importer

X, Y = data_importer.load_data(0)    
print("X", X.shape, "Y", Y.shape)

# Build neural network
net = tflearn.input_data(shape=[None, 2])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)

print("toto")
# Start training (apply gradient descent algorithm)
model.fit(X, Y, n_epoch=10, batch_size=1, show_metric=True)

