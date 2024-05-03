import deepxde as dde
import numpy as np

class PINN():
    
    def __init__(self, dynamics, heter, inverse):
        
        ## Dynamics
        self.dynamics = dynamics
        self.heter = heter
        self.inverse = inverse
        
        ## PDE Parameters (initialized for 1D PINN)
        self.input = 3 # network input size 
        self.num_hidden_layers = 5 # number of hidden layers for NN 
        self.hidden_layer_size = 60 # size of each hidden layers 
        self.output = 2 # network input size 
        
        ## Training Parameters
        self.num_domain = 40000 # number of training points within the domain
        self.num_boundary = 4000 # number of training boundary condition points on the geometry boundary
        self.num_test = 1000 # number of testing points within the domain
        self.MAX_MODEL_INIT = 16 # maximum number of times allowed to initialize the model
        self.MAX_LOSS = 4 # upper limit to the initialized loss
        self.epochs_init = 15000 # number of epochs for training initial phase
        self.epochs_main = 150000 # number of epochs for main training phase
        self.lr = 0.0005 # learning rate
        
        ## Update constants for inverse and/or heterogeneity geometry
        self.modify_const()
    
    def modify_const(self):
        ## Update the PINN design for inverse and/or heterogeneity geometry
        if self.heter:
            self.output = 3
        if self.inverse:
            self.lr = 0.0001
    
    def define_pinn(self, geomtime, input_data, observe_train):
        
        ## Define the network
        self.net = dde.maps.FNN([self.input] + [self.hidden_layer_size] * self.num_hidden_layers + [self.output], "tanh", "Glorot uniform")
        
        ## Select relevant PDE (Heterogeneity, forward/inverse)
        if self.heter:
            if self.inverse and 'd' in self.inverse:
                pde = self.dynamics.pde_2D_heter
                self.net.apply_output_transform(self.dynamics.modify_inv_heter)
            else:
                pde = self.dynamics.pde_2D_heter_forward
                self.net.apply_output_transform(self.dynamics.modify_heter)
        elif not self.heter:
            pde = self.dynamics.pde_2D     
        
        ## Define PINN model
        self.pde_data = dde.data.TimePDE(geomtime, pde, input_data,
                            num_domain = self.num_domain, 
                            num_boundary=self.num_boundary, 
                            anchors=observe_train,
                            num_test=self.num_test)    
        self.model = dde.Model(self.pde_data, self.net)
        self.model.compile("adam", lr=self.lr)
        return 0
        
    def stable_init(self):
        
        ## Stabalize initialization process by capping the losses
        losshistory, _ = self.model.train(epochs=1)
        initial_loss = max(losshistory.loss_train[0])
        num_init = 1
        while initial_loss>self.MAX_LOSS or np.isnan(initial_loss):
            num_init += 1
            self.model = dde.Model(self.pde_data, self.net)
            self.model.compile("adam", lr=self.lr, loss_weights = [0,0,0,0,1])
            losshistory, _ = self.model.train(epochs=1)
            initial_loss = max(losshistory.loss_train[0])
            if num_init > self.MAX_MODEL_INIT:
                raise ValueError('Model initialization phase exceeded the allowed limit')
        return 0
    
    def train(self, out_path, params):
        
        ## Stabalize initialization process by capping the losses
        self.stable_init()
        ## Train PINN with corresponding scheme
        losshistory, train_state = self.train_3_phase(out_path, params)
        return self.model, losshistory, train_state
    
    def train_3_phase(self, out_path, params):
        init_weights = [0,0,0,0,1]
        if self.inverse:
            variables_file = "variables_" + self.inverse + ".dat"
            variable = dde.callbacks.VariableValue(params, period=1000, filename=variables_file)    
            ## Initial phase
            self.model.compile("adam", lr=0.0005, loss_weights=init_weights)
            losshistory, train_state = self.model.train(epochs=self.epochs_init, model_save_path = out_path, callbacks=[variable])
            ## Main phase
            self.model.compile("adam", lr=self.lr)
            losshistory, train_state = self.model.train(epochs=self.epochs_main, model_save_path = out_path, callbacks=[variable])
            ## Final phase
            self.model.compile("L-BFGS-B")
            losshistory, train_state = self.model.train(model_save_path = out_path, callbacks=[variable])
        else:
            ## Initial phase
            self.model.compile("adam", lr=0.0005, loss_weights = init_weights)
            losshistory, train_state = self.model.train(epochs=self.epochs_init, model_save_path = out_path)
            ## Main phase
            self.model.compile("adam", lr=self.lr)
            losshistory, train_state = self.model.train(epochs=self.epochs_main, model_save_path = out_path)
            #losshistory, train_state = self.model.train(epochs=15000, model_save_path = out_path)
            ## Final phase
            self.model.compile("L-BFGS-B")
            losshistory, train_state = self.model.train(model_save_path = out_path)
        return losshistory, train_state
        
