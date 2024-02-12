import torch
import numpy as np

def WeightStand(w, eps=1e-5):
    mean = torch.mean(input=w, dim=[0,1], keepdim=True)
    var = torch.var(input=w, dim=[0,1], keepdim=True)
    w = (w - mean) / torch.sqrt(var + eps)

    return w

def WeightClipping(w):
    return torch.clamp(w, -3, 3)

class Neurons:
    def __init__(self, n_neurons, last=False):
        """
        A single layer of linear projection followed by the complex neuron activation function
        """
        self.n_neurons = n_neurons
        self.ones = torch.ones((n_neurons, 1), dtype=torch.float)
        self.hidden = torch.zeros((n_neurons, 1), dtype=torch.float)
        self.params = torch.rand(n_neurons, 3, 3, dtype=torch.float) * 0.2 - 0.1
        self.last = last

    def neuron_fn(self, inputs):
        """
        inputs: (n_neurons, )
        """

        assert inputs.shape == (self.n_neurons,), inputs.shape
        inputs = inputs[:,None]
        stacked = torch.hstack((inputs, self.ones, self.hidden))[:,:,None]

        if not self.last:
            dot = torch.tanh((self.params.to(torch.float) @ stacked.to(torch.float)).squeeze())
        elif self.last:
            dot = torch.tanh((self.params.to(torch.float) @ stacked)).reshape(1,3)

        x = dot[:,0]
        self.hidden = dot[:, -1][:, None]

        return x, dot


class HebbianNet:
    def __init__(self, sizes):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        self.weights = [torch.Tensor(sizes[i], sizes[i + 1]).uniform_(-0.1,0.1) for i in range(len(sizes) - 1)]

        self.A = [torch.normal(0,.1, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.B = [torch.normal(0,.1, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.C = [torch.normal(0,.1, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.D = [torch.normal(0,.1, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.lr = [torch.normal(0,.1, (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]

        neurons = [Neurons(size) for size in sizes[:-1]]
        neurons.append(Neurons(sizes[-1], last=False))
        self.neurons = neurons


    def forward(self, pre):

        with torch.no_grad():
            pre = torch.from_numpy(pre)
            """
            pre: (n_in, )
            """

            neurons = self.neurons
            weights = self.weights

            pre, pre_dot = neurons[0].neuron_fn(pre)
            
            for i, neuron in enumerate(neurons[1:]):
                    send = pre.to(torch.float) @ weights[i].to(torch.float)
                    post, post_dot = neuron.neuron_fn(send)

                    self.weights[i] = self.hebbian_update(weights[i], pre, post, self.A[i], self.B[i], self.C[i], self.D[i], self.lr[i])
                    pre = post
        
        return post.detach().numpy()


    def hebbian_update(self, weights, pre, post, A, B, C, D, lr):

        i = torch.ones(weights.shape) * pre.unsqueeze(1)
        j = torch.ones(weights.shape) * post
        ij = i * j

        weights = weights + lr * (A*ij + B*i + C*j + D)
        weights = WeightStand(weights)
        return weights


    def get_params(self):
        p1 = torch.cat([ params.flatten() for params in self.A]  
                +[ params.flatten() for params in self.B] 
                +[ params.flatten() for params in self.C]
                +[ params.flatten() for params in self.D]
                +[ params.flatten() for params in self.lr]
                )
        p2 = torch.cat([neurons.params for neurons in self.neurons])

        return torch.cat([p1.flatten(), p2.flatten()])
        
    def set_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)

        m = 0
        for i, hebb_A in enumerate(self.A):
            a, b = hebb_A.shape
            self.A[i] = flat_params[m:m + a * b].reshape(a, b)
            m += a * b 

        for i, hebb_B in enumerate(self.B):
            a, b = hebb_B.shape
            self.B[i] = flat_params[m:m + a * b].reshape(a, b)
            m += a * b 

        for i, hebb_C in enumerate(self.C):
            a, b = hebb_C.shape
            self.C[i] = flat_params[m:m + a * b].reshape(a, b)
            m += a * b 

        for i, hebb_D in enumerate(self.D):
            a, b = hebb_D.shape
            self.D[i] = flat_params[m:m + a * b].reshape(a, b)
            m += a * b 

        for i, hebb_lr in enumerate(self.lr):
            a, b = hebb_lr.shape
            self.lr[i] = flat_params[m:m + a * b].reshape(a, b)
            m += a * b

        for neuron in self.neurons:
            a, b, c = neuron.params.shape
            neuron.params = flat_params[m:m + a * b * c].reshape(a, b, c)
            m += a * b * c
