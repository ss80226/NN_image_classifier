
import imageio
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
np.random.seed(6666) # to reprodece the performance, you can also remove it
class twoLayerNN():
    
    def __init__(self, input_shape, output_shape, hidden_d=32):
        # seeding for random number generation
        np.random.seed(1)
        self.weights1   = np.random.rand(input_shape+1,hidden_d) # +1 is the bias 
        self.weights2   = np.random.rand(hidden_d,output_shape)
        self.b_std = np.zeros(shape=(hidden_d,))
        self.b_mean = np.zeros(shape=(hidden_d,))
        self.bias_1 = np.random.rand(1, hidden_d) # hidden layer1 bias (from [0.1))
        self.const = 0.00000000001 # prevent from / 0
        self.step = 0.1

    def weights(self): # testing for debug
        return self.weights1, self.weights2
    
    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1).reshape(x.shape[0], 1))
        return exps / np.sum(exps, axis=1).reshape(x.shape[0], 1)
    
    def batch_norm(self, x, bigBatch=True):
        if bigBatch: # training time
            m = np.mean(x, axis=0)
            s = np.std(x, axis=0)
        else: # testing time
            m = self.b_mean
            s = self.b_std
        return (x-m)/(s+self.const), m, s
    
    def batch_norm_derivative(self, x):
        return (np.std(x, axis=0) + self.const)**(-1)
    
    def softmax_cross_entropy_deriative(self, x, y):
        return (x - y) # see how cross entropy loss & softmax deriative 
    
    def cross_entropy(self, x, y):
        return -np.log(x+self.const)*y
    
    def forward(self, x):
        bias_0 = np.random.randn(x.shape[0], 1) # input bais
        x = np.concatenate((bias_0, x), axis=1) # concatenate the bias to the input
        z1 = np.dot(x, self.weights1)
        batch_norm1, _, _ = self.batch_norm(z1, bigBatch=False)
        layer1 = self.sigmoid(batch_norm1)
        bias_1 = self.bias_1 # hidden layer1 bias (from [0.1))
        layer1 += bias_1
        z2 = np.dot(layer1, self.weights2)
        out = self.softmax(z2)
        return out, z1, z2
    
    def backprop(self, x, y, iteration):
        bias_0 = np.random.randn(x.shape[0], 1) # input bais
        x = np.concatenate((bias_0, x), axis=1) # concatenate the bias to the input
        
        z1 = np.dot(x, self.weights1)
        batch_norm1, mean, std = self.batch_norm(z1)
        if not iteration == 0:
            self.b_mean = 0.9*self.b_mean + 0.1*mean
            self.b_std = 0.9*self.b_std + 0.1*std
        else:
            self.b_mean = mean
            self.b_std = std
        layer1 = self.sigmoid(batch_norm1)
        bias_1 = self.bias_1 # hidden layer1 bias (from [0.1))
        layer1_b = layer1+bias_1 # WX+b
        z2 = np.dot(layer1_b, self.weights2)
        out = self.softmax(z2)
        
        ## backprop & calculate the gradient
        
        d_softmax_cross = self.softmax_cross_entropy_deriative(out, y)/y.shape[0] # mean
        d_weights2 = np.dot(np.transpose(layer1_b), d_softmax_cross)
        b = np.dot(d_softmax_cross, np.transpose(self.weights2))         * self.sigmoid_derivative(layer1) * self.batch_norm_derivative(z1)

        d_weights1 = np.dot(np.transpose(x),  b)
        d_bias1 = np.dot(np.ones(shape=(1, x.shape[0])), b)

        ## update weights
        self.weights2 -= self.step*d_weights2
        self.weights1 -= self.step*d_weights1
        self.bias_1 -= self.step*d_bias1
        
        loss = self.cross_entropy(out, y)
        loss = np.sum(loss, axis=1)
        loss = np.mean(loss)
        accuracy = self.accuracy(out, y)
        return loss, accuracy
    
    def train(self, data, label ,total_iteration, batch_size=128, log=True):
        for i in range(total_iteration):
            idx = np.random.choice(490*3, batch_size, replace=False)
            loss, accuracy = self.backprop(data[idx], label[idx], i)
            if i%1000 == 0 and log is True:
                print("iteration %d: cross entrpy loss is %f, accuracy is %f" %(i, loss, accuracy))
        return loss
    def test(self, data, label, log=False):
        batch_size = data.shape[0]
        out, _, _ = self.forward(data)
        if data.shape[0] > 10: # if input test size is not too big, show the prediction
            accuracy = self.accuracy(out, label, log=log)
            print("testing size : %d, accuracy: %f" %(batch_size, accuracy))
        else:                  # otherwise, show the accuracy only
            prediction = np.argmax(out, axis=1)[0]
            ground_truth = np.argmax(label, axis=1)[0]
            print("prediction is : %d, ground truth is: %d" %(prediction, ground_truth))

    def accuracy(self, x, y, log=False):
        prediction = np.argmax(x, axis=1)
       
        ground_truth = np.argmax(y, axis=1)
        comparison = (prediction == ground_truth)
        r = np.sum(comparison) # count the number of correct  answer(true)
        if log:
            print(prediction)
            print(r)
        accuracy = r / y.shape[0] #accuracy
        return accuracy
class threeLayerNN():
    
    def __init__(self, input_shape, output_shape, hidden_d=32):
        # seeding for random number generation
        np.random.seed(1)
        self.weights1 = np.random.rand(input_shape+1,hidden_d) # +1 is the bias 
        self.weights2 = np.random.rand(hidden_d,hidden_d)
        self.weights3 = np.random.rand(hidden_d,output_shape)
        self.b_std = np.zeros(shape=(hidden_d,))
        self.b_mean = np.zeros(shape=(hidden_d,))
        self.b_std_1 = np.zeros(shape=(hidden_d,))
        self.b_mean_1 = np.zeros(shape=(hidden_d,))
        self.bias_1 = np.random.rand(1, hidden_d) # hidden layer1 bias (from [0.1))
        self.bias_2 = np.random.rand(1, hidden_d) # hidden layer2 bias (from [0.1))
        self.const = 0.00000000001 # prevent from / 0
        self.step = 0.1

    def weights(self):
        return self.weights1, self.weights2
    
    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1).reshape(x.shape[0], 1))
        return exps / np.sum(exps, axis=1).reshape(x.shape[0], 1)
    
    def batch_norm(self, x, bigBatch=True, second=False):
        if bigBatch:
            m = np.mean(x, axis=0)
            s = np.std(x, axis=0)
        else:
            if second is True: # 2nd layer batch norm
                m = self.b_mean_1
                s = self.b_std_1
            else:
                m = self.b_mean
                s = self.b_std
        return (x-m)/(s+self.const), m, s
    
    def batch_norm_derivative(self, x):
        return (np.std(x, axis=0) + self.const)**(-1)
    
    def softmax_cross_entropy_deriative(self, x, y):
        return (x - y) # see how cross entropy loss & softmax deriative 
    
    def cross_entropy(self, x, y):
        return -np.log(x+self.const)*y
    
    def forward(self, x):
        bias_0 = np.random.randn(x.shape[0], 1) # input bais
        x = np.concatenate((bias_0, x), axis=1) # concatenate the bias to the input
        z1 = np.dot(x, self.weights1)
        batch_norm1, _, _ = self.batch_norm(z1, bigBatch=False)
        layer1 = self.sigmoid(batch_norm1)
        bias_1 = self.bias_1 # hidden layer1 bias (from [0.1))
        layer1 += bias_1
        z2 = np.dot(layer1, self.weights2)
        batch_norm2, _, _ = self.batch_norm(z2, bigBatch=False, second=True)
        layer2 = self.sigmoid(batch_norm2)
        bias_2 = self.bias_2 # hidden layer1 bias (from [0.1))
        layer2 += bias_2
        z3 = np.dot(layer2, self.weights3)
        out = self.softmax(z3)
        return out, z1, z2, z3
    
    def backprop(self, x, y, iteration):
        bias_0 = np.random.randn(x.shape[0], 1) # input bais
        x = np.concatenate((bias_0, x), axis=1) # concatenate the bias to the input
        
        z1 = np.dot(x, self.weights1)
        batch_norm1, mean, std = self.batch_norm(z1)
#         print(mean.shape)
        if not iteration == 0:
            self.b_mean = 0.9*self.b_mean + 0.1*mean
            self.b_std = 0.9*self.b_std + 0.1*std
        else:
            self.b_mean = mean
            self.b_std = std
        layer1 = self.sigmoid(batch_norm1)
        
        bias_1 = self.bias_1 # hidden layer1 bias (from [0.1))
        layer1_b = layer1+bias_1 # WX+b
        z2 = np.dot(layer1_b, self.weights2)
        batch_norm2, mean, std = self.batch_norm(z2, second=True)
        if not iteration == 0:
            self.b_mean_1 = 0.9*self.b_mean_1 + 0.1*mean
            self.b_std_1 = 0.9*self.b_std_1 + 0.1*std
        else:
            self.b_mean = mean
            self.b_std = std
        layer2 = self.sigmoid(batch_norm2)
        
        bais_2 = self.bias_2
        layer2_b = layer2 + bais_2
        z3 = np.dot(layer2_b, self.weights3)
        out = self.softmax(z3)
        
#         print(layer1)
        ## backprop & calculate the gradient
        
        d_softmax_cross = self.softmax_cross_entropy_deriative(out, y)/y.shape[0] # mean
        d_weights3 = np.dot(np.transpose(layer2_b), d_softmax_cross)
        
        b0 = np.dot(d_softmax_cross, np.transpose(self.weights3))         * self.sigmoid_derivative(layer2) * self.batch_norm_derivative(z2)
        d_weights2 = np.dot(np.transpose(layer1_b), b0)
        d_bias2 = np.dot(np.ones(shape=(1, layer1_b.shape[0])), b0) 
        
        b = np.dot(b0, np.transpose(self.weights2))         * self.sigmoid_derivative(layer1) * self.batch_norm_derivative(z1)
        d_weights1 = np.dot(np.transpose(x),  b)
        d_bias1 = np.dot(np.ones(shape=(1, x.shape[0])), b)
        
        ## update weights
        self.weights3 -= self.step*d_weights3
        self.weights2 -= self.step*d_weights2
        self.weights1 -= self.step*d_weights1
        self.bias_2 -= self.step*d_bias2
        self.bias_1 -= self.step*d_bias1 
#         print(d_weights1)
        
        loss = self.cross_entropy(out, y)
        loss = np.sum(loss, axis=1)
        loss = np.mean(loss)
        accuracy = self.accuracy(out, y)
        return loss, accuracy
    
    def train(self, data, label ,total_iteration, batch_size=128, log=True):
        for i in range(total_iteration):
            idx = np.random.choice(490*3, batch_size, replace=False)
            loss, accuracy = self.backprop(data[idx], label[idx], i)
            if i%1000 == 0 and log is True:
                print("iteration %d: cross entrpy loss is %f, accuracy is %f" %(i, loss, accuracy))
        return loss
    def test(self, data, label, log=False):
        batch_size = data.shape[0]
        out, _, _, _ = self.forward(data)
        if data.shape[0] > 10: # if input test size is not too big, show the prediction
            accuracy = self.accuracy(out, label, log=log)
            print("testing size : %d, accuracy: %f" %(batch_size, accuracy))
        else:                  # otherwise, show the accuracy only
            prediction = np.argmax(out, axis=1)[0]
            ground_truth = np.argmax(label, axis=1)[0]
            print("prediction is : %d, ground truth is: %d" %(prediction, ground_truth))

    def accuracy(self, x, y, log=False):
        prediction = np.argmax(x, axis=1)
       
        ground_truth = np.argmax(y, axis=1)
        comparison = (prediction == ground_truth)
        r = np.sum(comparison) # count the number of correct  answer(true)
        if log:
            print(prediction)
            print(r)
        accuracy = r / y.shape[0] #accuracy
        return accuracy




def main():
    data = np.zeros(shape=(490*3, 32, 32, 4))
    label = np.zeros(shape=(490*3, 3)) # one hot 
    for i, im_path in enumerate(glob.glob("./Data/Data_train/Carambula/*.png")):
        im = imageio.imread(im_path)
        data[i] = np.array(im)
        label[i] = np.array([1, 0, 0]) # one hot
    for i, im_path in enumerate(glob.glob("./Data/Data_train/Lychee/*.png")):
        im = imageio.imread(im_path)
        data[490+i] = np.array(im)
        label[490+i] = np.array([0, 1, 0]) # one hot
    for i, im_path in enumerate(glob.glob("./Data/Data_train/Pear/*.png")):
        im = imageio.imread(im_path)
        data[490*2+i] = np.array(im)
        label[490*2+i] = np.array([0, 0, 1])
    d = data.flatten().reshape(490*3, 4096) # flatten & reshape the data array

    test_data = np.zeros(shape=(166*3, 32, 32, 4))
    test_label = np.zeros(shape=(166*3, 3)) # one hot 
    for i, im_path in enumerate(glob.glob("./Data/Data_test/Carambula/*.png")):
        im = imageio.imread(im_path)
        test_data[i] = np.array(im)
        test_label[i] = np.array([1, 0, 0]) # one hot
    for i, im_path in enumerate(glob.glob("./Data/Data_test/Lychee/*.png")):
        im = imageio.imread(im_path)
        test_data[166+i] = np.array(im)
        test_label[166+i] = np.array([0, 1, 0]) # one hot
    for i, im_path in enumerate(glob.glob("./Data/Data_test/Pear/*.png")):
        im = imageio.imread(im_path)
        test_data[166*2+i] = np.array(im)
        test_label[166*2+i] = np.array([0, 0, 1])
        test_d = test_data.flatten().reshape(166*3, 4096) # flatten & reshape the data array

    # ### Standardize the Data
    # Normalize the input data, since PCA is effected by scale
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(d)
    # Apply transform to both the training set and the test set.

    train_img = scaler.transform(d)
    test_img = scaler.transform(test_d)
    pca = PCA(n_components=2)

    pca.fit(train_img)

    train_img = pca.transform(train_img)
    test_img = pca.transform(test_img)


    nn = twoLayerNN(train_img.shape[1], 3, hidden_d=32)
    print('\n')
    print("===================Training 2-layer NN==================")
    nn.train(train_img, label, 6000)
    print("========================================================\n")
    print("===================Testing 2-layer NN===================")
    nn.test(test_img, test_label, log=False)
    print("========================================================\n")
    nn3 = threeLayerNN(train_img.shape[1], 3, hidden_d=32)
    print("===================Training 3-layer NN==================")
    nn3.train(train_img, label, 6000)
    print("========================================================\n")
    print("===================Testing 3-layer NN===================")
    nn3.test(test_img, test_label, log=False)


if __name__ == '__main__':
    main()