#
# Lesson #1 challenge from Sirajology - Introduction - Learn Python for Data Science #1
# https://www.youtube.com/watch?v=T5pRlIbr6gg
# Ported to tf learn
#
# Using CDC BRFSS Annual Survey Data 2015 
# https://www.cdc.gov/brfss/annual_data/annual_2015.html

# (C) 2016 - Shazz 
# Under MIT license

# classifier
import tflearn
import data_importer
import numpy as np

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

# Start training (apply gradient descent algorithm)
model.fit(X, Y, n_epoch=4, batch_size=200, show_metric=True, validation_set=0.3)

pred = model.predict([[83, 183], [40, 144]])
print("a person of 83 kg 183 cm has a probability of", pred[0][1]*100, "% to be a man")
print("a person of 40 kg 144 cm has a probability of", pred[1][1]*100, "% to be a man")


