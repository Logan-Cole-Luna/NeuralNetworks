import numpy as np
from BackPropNetwork import relu, tanh, NeuralNet, mean_squared_error
# Create a neural network for the XOR logic gate, so two inputs and one output neuron. 

#logic gate inputs (this will be our training set)
inputs  = np.array([[0,0], 
                    [0,1], 
                    [1,0], 
                    [1,1]])

#outputs of different logic gates when given the inputs above
training_targets = {
    'XOR'  : np.array([[0],
                       [1],
                       [1],
                       [0]]),

    'OR'   : np.array([[0],
                       [1],
                       [1],
                       [1]]),

    'AND'  : np.array([[0],
                       [0],
                       [0],
                       [1]]),

    'NAND' : np.array([[1],
                       [1],
                       [1],
                       [0]]),
}

#-Network parameters
m           = 0.9            #momentum
a           = 0.01           #learning rate
init_method = 'xavier'       #weight initialization method
hidden_f    = relu           #activation function for hidden layers
output_f    = tanh           #activation function for output layer
epochs      = 1000
batch_size  = 0
e_threshold = 1E-5           #error threshold to stop training

# n1 - Number of input nuerons, must be equal to inputs, in this case training set
# n2 - Number of hidden neurons
# n3 - Number of output neurons, must be equal to output, in this case answer to set
network_topology = [2, 10, 8, 1] # Example: 2 inputs, two hidden layers (10 and 8 neurons), 1 output

for gate in training_targets:
    print(f"\n\n{'='*40}\nTraining {gate} gate:\n{'='*40}\n")
    input_data = inputs 
    target_data = training_targets[gate]

    #-- create a new network for each set so they are independent of each other
    nnet = NeuralNet(network_topology, hidden_activation_func=hidden_f, output_activation_func=output_f, init_method=init_method, momentum=m, learning_rate=a)
    error = nnet.train(input_data, target_data, epochs=epochs, error_func=mean_squared_error, batch_size=batch_size, error_threshold=e_threshold)

    # Test different inputs
    for i, sample in enumerate(input_data):
        output = nnet.feedforward(sample)
        sample_error = mean_squared_error(target_data[i:i+1], output)
        print(f"Testing Network:\n\tinput vector    : {sample}\n\toutput vector   : {output}\n\texpected output : {target_data[i]}")
        print(f"\tNetwork error   : {sample_error:.3e}\n")