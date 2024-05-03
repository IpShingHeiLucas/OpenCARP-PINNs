import os, sys, subprocess, re, struct, errno
import scipy.io
import deepxde as dde
from deepxde.backend import tf
import numpy as np



class system_dynamics():
    
    def __init__(self):
        
        ## PDE Parameters
        self.a = 0.15
        self.b = 0.15
        self.D = 0.05
        self.k = 8
        self.mu_1 = 0.2 #Can be obtained form the OpenCARP parameter file
        self.mu_2 = 0.3 #Can be obtained form the OpenCARP parameter file
        self.epsilon = 0.002
        self.touAcm2 = 100/12.9
        self.t_norm = 12.9

        ## Geometry Parameters
        self.min_x = 0
        self.max_x = 10            
        self.min_y = 0 
        self.max_y = 10
        self.min_t = 0
        self.max_t = 99
        self.spacing = 0.1

    def read_array_igb(self, igbfile):
        """
        Purpose: Function to read a .igb file
        """
        data = []
        file = open(igbfile, mode="rb")
        header = file.read(1024)
        words = header.split()
        word = []
        for i in range(4):
            word.append(int([re.split(r"(\d+)", s.decode("utf-8")) for s in [words[i]]][0][1]))

        nnode = word[0] * word[1] * word[2]

        for _ in range(os.path.getsize(igbfile) // 4 // nnode):
            data.append(struct.unpack("f" * nnode, file.read(4 * nnode)))

        file.close()
        return data

    def read_pts(self, modname, n=3, vtx=False, item_type=float):
        """Read pts/vertex file"""
        with open(modname + (".vtx" if vtx else ".pts")) as file:
            count = int(file.readline().split()[0])
            if vtx:
                file.readline()

            pts = np.empty((count, n), item_type)
            for i in range(count):
                pts[i] = [item_type(val) for val in file.readline().split()[0:n]]

        return pts if n > 1 else pts.flat

    def generate_data(self, v_file_name, w_file_name, pt_file_name): #Temporary becasue we dont have W yet!!Plz add w_file_name later
        
        data_V = np.array(self.read_array_igb(v_file_name)) #new parser for vm.igb for voltage
        data_W = np.array(self.read_array_igb(w_file_name)) #new parser for vm.igb for W #Temporary becasue we dont have W yet!!
        coordinates = np.array(self.read_pts(pt_file_name)) #new parser for .pt file


        #t = np.arange(0, data_V.shape[0]/100, 0.01).reshape(-1, 1)
        t = np.arange(0, data_V.shape[0]).reshape(-1, 1)
        coordinates = (coordinates - np.min(coordinates))/1000
        coordinates = coordinates[:, 0:2]
        x = np.unique(coordinates[:, 0]).reshape((1, -1))
        y = np.unique(coordinates[:, 1]).reshape((1, -1))
        len_x = x.shape[1]
        len_y = y.shape[1]
        len_t = t.shape[0]

        no_of_nodes = coordinates.shape[0]
        repeated_array = np.repeat(coordinates, len_t, axis=0)
        xy_concatenate = np.vstack(repeated_array)
        t_concatenate = np.concatenate([t] * no_of_nodes, axis=0)
        grid = np.concatenate([xy_concatenate, t_concatenate], axis=1)

        data_V = (data_V + 80)/100
        data_W = (data_W + 80)/100
        data_V = data_V.T
        data_W = data_W.T

        shape = [len_x, len_y, len_t]
        V = data_V.reshape(-1, 1)
        W = data_W.reshape(-1, 1)

        shape = [len_x, len_y, len_t]
        Vsav = V.reshape(len_x, len_y, len_t)

        Wsav = W.reshape(len_x, len_y, len_t)

        ##Computing in Cardiology Extrapolation from source
        ##Corner
        #midpt_x = np.max(grid[:,0])*0.5
        #midpt_y = np.max(grid[:,1])*0.5
        #idx_data_smaller = np.where((grid[:,0]<=midpt_x) & (grid[:,1]<=midpt_y))
        #idx_data_larger = np.where((grid[:,0]>midpt_x) | (grid[:,1]>midpt_y))

        ##Planar & Double Corner
        first_quarter_x = np.max(grid[:,0])*0.25
        idx_data_smaller = np.where((grid[:,0]<=first_quarter_x))
        idx_data_larger = np.where((grid[:,0]>first_quarter_x))


        ##Computing in Cardiology Inverse Extrapolation
        #first_quat_x = np.max(grid[:,0])*0.25
        #first_quat_y = np.max(grid[:,1])*0.25
        #third_quat_x = np.max(grid[:,0])*0.75
        #third_quat_y = np.max(grid[:,1])*0.75

        #idx_data_smaller = np.where((grid[:,0]>=first_quat_x) & (grid[:,0]<=third_quat_x) & (grid[:,1]>=first_quat_y) & (grid[:,1]<=third_quat_y))
        #idx_data_larger = np.where((grid[:,0]<first_quat_x) | (grid[:,0]>third_quat_x) | (grid[:,1]<first_quat_y) | (grid[:,1]>third_quat_y))


        #The lower quadrant   
        smaller_grid = grid[idx_data_smaller]
        smaller_V = V[idx_data_smaller]
        smaller_W = W[idx_data_smaller]

        #The other 3 quadrant   
        larger_grid = grid[idx_data_larger]
        larger_V = V[idx_data_larger]
        larger_W = W[idx_data_larger]

        #Shuffling the data
        def shiffling(grid, V, W):
            num_rows = grid.shape[0]
            indices = np.arange(num_rows)
            np.random.shuffle(indices)
            
            grid = grid[indices]
            V = V[indices]
            W = W[indices]
            
            return grid, V, W

        observe_train, v_train, w_train = shiffling(smaller_grid, smaller_V, smaller_W)
        observe_test, v_test, w_test = shiffling(larger_grid, larger_V, larger_W)

        return observe_train, observe_test, v_train, v_test, w_train, w_test, Vsav, V, len_t, idx_data_larger, Wsav, W

    def geometry_time(self):  
        geom = dde.geometry.Rectangle([self.min_x,self.min_y], [self.max_x,self.max_y])
        timedomain = dde.geometry.TimeDomain(self.min_t, self.max_t)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        return geomtime

    def params_to_inverse(self,args_param):
        
        params = []
        if not args_param:
            return self.a, self.b, self.D, params
        ## If inverse:
        ## The tf.variables are initialized with a positive scalar, relatively close to their ground truth values
        if 'a' in args_param:
            self.a = tf.math.exp(tf.Variable(-3.92))
            params.append(self.a)
        if 'b' in args_param:
            self.b = tf.math.exp(tf.Variable(-1.2))
            params.append(self.b)
        if 'd' in args_param:
            self.D = tf.math.exp(tf.Variable(-1.6))
            params.append(self.D)
        return params
    
    def pde_2D(self, x, y):
    
        V, W = y[:, 0:1], y[:, 1:2]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  self.D*(dv_dxx + dv_dyy) + (self.k*V*(V-self.a)*(V-1) +W*V)
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        #eq_a = dv_dt -  self.D*(dv_dxx + dv_dyy) + (self.k*V*(V-self.a)*(V-1) +W*V)*self.touAcm2 
        #eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*((-W -self.k*V*(V-self.b-1))/self.t_norm)
        return [eq_a, eq_b]

    def pde_2D_heter(self, x, y):
    
        V, W, var = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        dv_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dv_dy = dde.grad.jacobian(y, x, i=0, j=1)
        
        ## Heterogeneity
        D_heter = tf.math.sigmoid(var)*0.08+0.02;
        dD_dx = dde.grad.jacobian(D_heter, x, i=0, j=0)
        dD_dy = dde.grad.jacobian(D_heter, x, i=0, j=1)
        
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  D_heter*(dv_dxx + dv_dyy) -dD_dx*dv_dx -dD_dy*dv_dy + self.k*V*(V-self.a)*(V-1) +W*V 
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]
 
    def pde_2D_heter_forward(self, x, y):
                
        V, W, D = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        dv_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dv_dy = dde.grad.jacobian(y, x, i=0, j=1)
        
        ## Heterogeneity
        dD_dx = dde.grad.jacobian(D, x, i=0, j=0)
        dD_dy = dde.grad.jacobian(D, x, i=0, j=1)
        
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  D*(dv_dxx + dv_dyy) -dD_dx*dv_dx -dD_dy*dv_dy + self.k*V*(V-self.a)*(V-1) +W*V 
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]   
 
    def IC_func(self,observe_train, v_train):
        
        T_ic = observe_train[:,2].reshape(-1,1)
        idx_init = np.where(np.isclose(T_ic,0))[0]
        v_init = v_train[idx_init]
        observe_init = observe_train[idx_init]
        return dde.PointSetBC(observe_init,v_init,component=0)
    
    def BC_func(self, geomtime):
        bc = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), self.boundary_func_2d, component=0)
        return bc
    
    def boundary_func_2d(self,x, on_boundary):
            return on_boundary and ~(x[0:2]==[self.min_x,self.min_y]).all() and  ~(x[0:2]==[self.min_x,self.max_y]).all() and ~(x[0:2]==[self.max_x,self.min_y]).all()  and  ~(x[0:2]==[self.max_x,self.max_y]).all() 
   
    def modify_inv_heter(self, x, y):                
        domain_space = x[:,0:2]
        D = tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(domain_space, 60,
                            tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 1, activation=None)        
        return tf.concat((y[:,0:2],D), axis=1)    
    
    def modify_heter(self, x, y):
        
        x_space, y_space = x[:, 0:1], x[:, 1:2]
        
        x_upper = tf.less_equal(x_space, 54*0.1)
        x_lower = tf.greater(x_space,32*0.1)
        cond_1 = tf.logical_and(x_upper, x_lower)
        
        y_upper = tf.less_equal(y_space, 54*0.1)
        y_lower = tf.greater(y_space,32*0.1)
        cond_2 = tf.logical_and(y_upper, y_lower)
        
        D0 = tf.ones_like(x_space)*0.02 
        D1 = tf.ones_like(x_space)*0.1
        D = tf.where(tf.logical_and(cond_1, cond_2),D0,D1)
        return tf.concat((y[:,0:2],D), axis=1)

