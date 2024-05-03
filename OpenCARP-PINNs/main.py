import sys
import os         
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import deepxde as dde # version 0.11 or higher
from generate_plot import plot_results
import utils
import pinn
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os         
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-vf', '--v-file-name', dest='v_file_name', required = True, type = str, help='igb File name for input voltage data')
    parser.add_argument('-wf', '--w-file-name', dest='w_file_name', required = True, type = str, help='igb File name for W input data') #Temporary becasue we dont have W yet!!
    parser.add_argument('-ptf', '--pt-file-name', dest='pt_file_name', required = True, type = str, help='pt File name for coordinates of input nodes, but please do not put .pts at the end')
    parser.add_argument('-m', '--model-folder-name', dest='model_folder_name', required = True, type = str, help='Folder name to save model (prefix /)')
    parser.add_argument('-n', '--noise', dest='noise', action='store_true', help='Add noise to the data')
    parser.add_argument('-w', '--w-input', dest='w_input', action='store_true', help='Add W to the model input data')
    parser.add_argument('-v', '--inverse', dest='inverse', required = False, type = str, help='Solve the inverse problem, specify variables to predict (e.g. a / ad / abd')
    parser.add_argument('-ht', '--heter', dest='heter', required = False, action='store_true', help='Predict heterogeneity - only in 2D')    
    parser.add_argument('-p', '--plot', dest='plot', required = False, action='store_true', help='Create and save plots')
    parser.add_argument('-a', '--animation', dest='animation', required = False, action='store_true', help='Create and save 2D Animation')
    args = parser.parse_args()

## General Params
noise = 0.1 # noise factor
test_size = 0.9 # precentage of testing data

def main(args):
    
    ## Get Dynamics Class
    dynamics = utils.system_dynamics()
    
    ## Parameters to inverse (if needed)
    params = dynamics.params_to_inverse(args.inverse)
    
    ## Generate Data 
    v_file_name = args.v_file_name
    w_file_name = args.w_file_name #Temporary becasue we dont have W yet!!
    pt_file_name = args.pt_file_name  
    observe_train, observe_test, v_train, v_test, w_train, w_test, Vsav, V, len_t, idx_data_larger, Wsav, W = dynamics.generate_data(v_file_name, w_file_name, pt_file_name)  #Temporary becasue we dont have W yet!! Plz add w_file_name later

    
    ## Add noise to training data if needed
    if args.noise:
        v_train = v_train + noise*np.random.randn(v_train.shape[0], v_train.shape[1])

    ## Geometry and Time domains
    geomtime = dynamics.geometry_time()
    ## Define Boundary Conditions
    bc = dynamics.BC_func(geomtime)
    ## Define Initial Conditions
    ic = dynamics.IC_func(observe_train, v_train)
    
    ## Model observed data
    observe_v = dde.PointSetBC(observe_train, v_train, component=0)
    input_data = [bc, ic, observe_v]
    if args.w_input: ## If W required as an input
        observe_w = dde.PointSetBC(observe_train, w_train, component=1)
        input_data = [bc, ic, observe_v, observe_w]
    
    ## Select relevant PDE (Heterogeneity) and define the Network
    model_pinn = pinn.PINN(dynamics,args.heter, args.inverse)
    model_pinn.define_pinn(geomtime, input_data, observe_train)
            
    ## Train Network
    out_path = dir_path + args.model_folder_name
    model, losshistory, train_state = model_pinn.train(out_path, params)
    
    ## Compute rMSE
    pred = model.predict(observe_test)   
    v_pred, w_pred = pred[:,0:1], pred[:,1:2]
    rmse_v = np.sqrt(np.square(v_pred - v_test).mean())
    print('--------------------------')
    print("V rMSE for test data:", rmse_v)
    print('--------------------------')
    print("Arguments: ", args)
    
    ## Save predictions, data
    np.savetxt("train_data.dat", np.hstack((observe_train, v_train, w_train)),header="observe_train,v_train, w_train")
    np.savetxt("test_pred_data.dat", np.hstack((observe_test, v_test,v_pred, w_test, w_pred)),header="observe_test,v_test, v_pred, w_test, w_pred")


    file_name = args.model_folder_name + "_W"

    plot_results(Wsav, W, observe_train, w_train, observe_test, w_pred, len_t, idx_data_larger, file_name, False)
    plot_results(Vsav, V, observe_train, v_train, observe_test, v_pred, len_t, idx_data_larger, args.model_folder_name, args.animation)

## Run main code
model = main(args)
