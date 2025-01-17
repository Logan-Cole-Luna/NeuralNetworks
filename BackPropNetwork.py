import numpy as np

def tanh(x, derivative=False, ):
    '''Tanh function & derivative'''
    
    if derivative:
        tanh_val = np.tanh(x)
        # return 1.0 - x**2
        return 1.0 - tanh_val**2
    else:
        # TODO create my own tanh
        return np.tanh(x)
    
def relu(x, derivative = False):
    if derivative:
        # returns 1 for any x > 0, otherwise 0
        return 1 * (x > 0)
    return np.maximum(0, x)

def mean_squared_error(target_output, actual_output, derivative = False):
    try:
        assert(target_output.shape == actual_output.shape)
    except AssertionError:
        print(f"Shape of target vector: {target_output.shape} does not match shape of actual vector: {actual_output.shape}")
        raise
    if derivative:
        error = (actual_output - target_output)
    else:
        # Changed error calculation to be more stable
        error = np.mean(np.square(target_output - actual_output))
    return error

class NeuralNet(object):
    RNG = np.random.default_rng()
    
    def __init__(self,
                 topology:list[int] = [], 
                 learning_rate = 0.01,
                 momentum = 0.1,
                 hidden_activation_func=relu,
                 output_activation_func=tanh,
                 init_method='random'):
        self.topology = topology
        self.weight_mats = []
        # Hold weights for bias nodes
        self.bias_mats = []
        
        self.hidden_activation = hidden_activation_func
        self.output_activation = output_activation_func
        
        self.learning_rate = learning_rate
        self.momentum = momentum

        self._init_weights_and_biases(init_method)
        self.size = len(self.weight_mats)
        self.netIns = [None] * self.size
        self.netOuts = [None]*self.size
        self.stored_gradients = [None] * self.size
        self.last_change = [np.zeros(mat.shape) for mat in self.weight_mats]
        
        #-- create similar lists to store gradients for the bias weigths
        #self.stored_bias_gradients = [np.zeros(mat.shape) for mat in self.bias_mats]
        self.last_bias_change      = [np.zeros(mat.shape) for mat in self.bias_mats]

    def _init_weights_and_biases(self, method='random'):
        if method.lower() == 'random':
            _init_func = lambda num_rows, num_cols: self.RNG.random(size=(num_rows, num_cols))
        elif method.lower() == 'xavier':
            _init_func = self._xavier_weight_initialization
        else:
            print(f"\t-> initialization method {method} not recognized. Defaulting to 'random'")
            _init_func = lambda num_rows, num_cols: self.RNG.random(size=(num_rows, num_cols))

        # Initialize weights between layers
        if len(self.topology) > 1:
            for layer_idx in range(len(self.topology) - 1):
                num_rows = self.topology[layer_idx]     # neurons in current layer
                num_cols = self.topology[layer_idx + 1] # neurons in next layer
                
                # Initialize weights and biases for this layer
                mat = _init_func(num_rows, num_cols)  
                bias_vector = _init_func(1, num_cols) 
                
                self.weight_mats.append(mat)
                self.bias_mats.append(bias_vector)

    def _xavier_weight_initialization(self, num_rows, num_cols):
        '''A type of weight initialization that seems to be tailored to sigmoidal activation functions.
        Here is a reference: https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/'''
        num_inputs = self.topology[0]

        lower_bound = -1 / np.sqrt(num_inputs)
        upper_bound = 1 / np.sqrt(num_inputs)

        mat = self.RNG.uniform(lower_bound, upper_bound, (num_rows, num_cols))
        return mat

    @property
    def shape(self):
        return tuple(self.topology)
    
    @property
    def n_trainable_params(self):
        n_params = 0 
        for weight_mat, bias_mat in zip(self.weight_mats, self.bias_mats):
            n_params += weight_mat.size + bias_mat.size

        return n_params
        


    def feedforward(self, input_vector):

        self.netIns.clear()
        self.netOuts.clear()

        I = input_vector  #rename vector to match typical nomenclature

        for idx, W in enumerate(self.weight_mats):
            
            bias_vector = self.bias_mats[idx]

            self.netOuts.append(I)    #storing activations from the last layer
            I = np.dot(I, W) + bias_vector
            self.netIns.append(I)       #storing the inputs to the current layer
            
            #-- apply activation function
            if idx == len(self.weight_mats) - 1:
                out_vector = self.output_activation(I)  #output layer
            else:
                I          = self.hidden_activation(I)  #hidden layers
            
        return out_vector
    

    def _gradient_descent(self, layer_idx, gradient_mat, bias_gradient):
        """Improved gradient descent with gradient clipping"""
        # Clip gradients to prevent explosion
        max_grad_norm = 1.0
        gradient_mat = np.clip(gradient_mat, -max_grad_norm, max_grad_norm)
        bias_gradient = np.clip(bias_gradient, -max_grad_norm, max_grad_norm)
        
        # Calculate weight updates
        delta_weight = (self.momentum * self.last_change[layer_idx]) - (self.learning_rate * gradient_mat)
        mean_bias_gradient = np.mean(bias_gradient, axis=0, keepdims=True)
        delta_bias_weights = (self.momentum * self.last_bias_change[layer_idx]) - (self.learning_rate * mean_bias_gradient)
        
        # Update weights with gradient clipping
        self.weight_mats[layer_idx] += delta_weight
        self.bias_mats[layer_idx] += delta_bias_weights
        
        # Store updates for momentum
        self.last_change[layer_idx] = delta_weight
        self.last_bias_change[layer_idx] = delta_bias_weights

    def backprop(self, 
                 target,
                 output,
                 error_func,):
        """Backpropagation.

        Parameters
        ----------
        target : numpy array
            Matching targets for each sample in `input_samples`.
        output : numpy array
            Actual output from feedforward propagation. It will be used to check the network's error.
        error_func : function object
            This is the function that computes the error of the epoch and used during backpropagation.
            It must accept parameters as: error_func(target={target numpy array},actual={actual output from network},derivative={boolean to indicate operation mode})
        """

        #Compute gradients and deltas
        for i in range(self.size):
            back_index =self.size-1 -i                  # This will be used for the items to be accessed backwards

            if i == 0:   #final layer
                d_activ = self.output_activation(self.netIns[back_index], derivative=True)
                d_error = error_func(target,output,derivative=True)
                delta = d_error * d_activ   #this should be the hadamard product, I think
                #delta = np.multiply(d_error, d_activ)

                gradient_mat  = np.dot(self.netOuts[back_index].T , delta)
                bias_grad_mat = 1 * delta

                #-- Apply gradient descent
                self._gradient_descent(layer_idx=back_index, gradient_mat=gradient_mat, bias_gradient=bias_grad_mat)

            else:     #hidden layers
                W_trans = self.weight_mats[back_index+1].T        #we use the transpose of the weights in the current layer
                d_activ = self.hidden_activation(self.netIns[back_index],derivative=True)  #δl=((wl+1)Tδl+1)⊙σ′(zl)
                d_error = np.dot(delta, W_trans)
                delta = d_error * d_activ   #this should be the hadamard product, I think
                #delta = np.multiply(d_error, d_activ)

                gradient_mat = np.dot(self.netOuts[back_index].T , delta)
                bias_grad_mat = 1 * delta

                #-- Apply gradient descent
                self._gradient_descent(layer_idx=back_index, gradient_mat=gradient_mat, bias_gradient=bias_grad_mat)


    def train(self, input_set, target_set, epochs=1000, batch_size=0, error_threshold=1E-10, error_func=mean_squared_error, verbose=True):

        if batch_size == 0:     #online training (one sample at a time)
            
            for epoch in range(epochs):
                error = 0

                for i in range(len(input_set)):
                    inputs = input_set[i:i+1]   #slicing it this way makes sure that the resulting numpy array maintains all of its dimensions
                    targets = target_set[i:i+1]

                    error += self._train_helper(inputs, targets, error_func)

                if verbose and (epoch % 250 == 0):
                    self._print_training_info(epoch, epochs, error, error_threshold)

                if error <= error_threshold:
                    print(f"\t-> error {error} is lower than threshold {error_threshold}\n\tStopped at epoch {epoch}")
                    break

        elif batch_size == -1:     #batch training (use full training set)
            
            for epoch in range(epochs):
                error = 0

                inputs  = input_set
                targets = target_set

                error += self._train_helper(inputs, targets, error_func)
                if verbose and (epoch % 20 == 0):
                    self._print_training_info(epoch, epochs, error, error_threshold)
                
                if error <= error_threshold:
                        print(f"\t-> error {error} is lower than threshold {error_threshold}\n\tStopped at epoch {epoch}")
                        break
                
        else:   #handle mini-batches later
            print("\t-> PROBLEM: mini-batches not supported yet. Choose batch_size 0 or -1")

        return error

    def _print_training_info(self, curr_epoch, total_epochs, curr_error, error_threshold):
        text = f"""{'-'*45}\n\t-> training step: :{curr_epoch}/{total_epochs}\n\t\t* current error: {curr_error}, threshold: {error_threshold}\n"""
        print(text)

    def _train_helper(self, input_set, target_set, error_func):
        nnet_output = self.feedforward(input_set)
        error       = error_func(target_set, nnet_output)
        
        self.backprop(target=target_set, output=nnet_output, error_func=error_func,)
        return error
