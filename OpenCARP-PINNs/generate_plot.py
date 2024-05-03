import sys
import os         
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

def plot_results(Vsav, V, observe_train, v_train, observe_test, v_pred, all_t, idx_data_larger, fig_name, animation):
    ##Formating Prediction data
    v_pred_train = np.concatenate((observe_train, v_train), axis=1)
    v_pred_test = np.concatenate((observe_test, v_pred), axis=1)
    v_pred_min = np.min(v_pred)
    v_pred_max = np.max(v_pred)

    stacked_array = np.vstack((v_pred_train, v_pred_test)) #Combining the training & testing data into one array
    sorted_array = stacked_array[np.lexsort((stacked_array[:, 2], stacked_array[:, 0], stacked_array[:, 1]))] #Rearranging the array's order based on x,y and t
    V_pred = np.array(sorted_array[:, 3],) #Only include the arrray of V in the correct order 
    Vsav_pred = V_pred.reshape(Vsav.shape)

    ##Create & Formating Observe data
    non_obv_array = np.full((observe_test.shape[0], 1), 999) #For later indication of non-observed data point
    v_obv_train_obv = np.concatenate((observe_train, v_train), axis=1) 
    v_non_obv_array = np.concatenate((observe_test, non_obv_array), axis=1)
    stacked_array_obv = np.vstack((v_obv_train_obv, v_non_obv_array))
    sorted_array_obv = stacked_array_obv[np.lexsort((stacked_array_obv[:, 2], stacked_array_obv[:, 0], stacked_array_obv[:, 1]))]
    V_obv = np.array(sorted_array_obv[:, 3],)
    Vsav_obv = V_obv.reshape(Vsav.shape)

    plot_action_pontential(Vsav, Vsav_pred, Vsav_obv, all_t, fig_name)
    plot_snapshot(Vsav_pred, v_pred_max, v_pred_min, fig_name)
    dice_score_calculator(V, V_pred, idx_data_larger)
    if animation:
        plot_animation(Vsav, Vsav_pred, v_pred_max, v_pred_min, fig_name)
    return 0

def plot_animation(Vsav, Vsav_pred, v_pred_max, v_pred_min, fig_name):
    def update_frame(frame):
        plt.clf()  # Clear the current figure

        # Plot for Vsav (Ground truth)
        ax1 = plt.subplot(121)
        im1 = ax1.imshow(Vsav[:, :, frame], cmap='jet', vmin=v_pred_min, vmax=v_pred_max, origin='lower')  # Transpose Vsav and display the current frame with fixed color scale
        plt.title('Ground Truth')

        # Plot for Vsav_pred (Prediction)
        ax2 = plt.subplot(122)
        im2 = ax2.imshow(Vsav_pred[:, :, frame], cmap='jet', vmin=v_pred_min, vmax=v_pred_max, origin='lower')  # Transpose Pred_V and display the current frame with fixed color scale
        plt.title('Prediction')

        # Create a common colorbar
        fig.subplots_adjust(right=0.8)  # Adjust the right margin to make space for the colorbar
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # Define the position of the colorbar
        fig.colorbar(im1, cax=cbar_ax)  # Add the colorbar to the figure
        cbar_ax.set_ylabel('V')

        # Add a common title showing the current time
        plt.suptitle(f'Time {frame*10}ms')  

    fig = plt.figure()
    fig.tight_layout()

    frames_per_second = 10  # Number of frames per second
    total_seconds = 500  # Total duration of the animation in seconds
    num_frames = frames_per_second * total_seconds  # Calculate the total number of frames

    anim = animation.FuncAnimation(fig, update_frame, frames=num_frames, interval=1000/frames_per_second)
    anim.save(fig_name+'_2D_Animation.mp4', writer=animation.FFMpegWriter(fps=10))
    plt.close(fig)

def plot_action_pontential(Vsav, Vsav_pred, Vsav_obv, all_t, fig_name):
    ## Get data
    cell_x = math.floor(len(Vsav[:, 0, 0]) * 0.75)
    cell_y = math.floor(len(Vsav[0, :, 0]) * 0.75)
    x = range(all_t)

    ## Create figure
    plt.plot(x, Vsav[cell_x, cell_y, :], label='Ground Truth')
    plt.plot(x, Vsav_pred[cell_x, cell_y, :], label='Prediction')
    observed_data = np.where(Vsav_obv[cell_x, cell_y, :] != 999)
    plt.plot(np.array(x)[observed_data], Vsav_obv[cell_x, cell_y, :][observed_data], 'r+', label='Observed')
    plt.xlabel('Time')
    plt.ylabel('V')
    plt.legend()

    ## Save figure
    plt.title(f'Action Potential at ({cell_x}, {cell_y}) node')
    plt.savefig(fig_name+'_Action_Potential.tiff', format='tiff', dpi=500)
    plt.close()

def plot_snapshot(Vsav_pred, v_pred_max, v_pred_min, fig_name):

    time_frame = math.floor(len(Vsav_pred[0, 0, :]) * 0.65)
    Vsav_pred_data_at_time_frame = Vsav_pred[:, :, time_frame].T

    # Generate x and y values
    x = np.arange(len(Vsav_pred[:, 0, 0]))
    y = np.arange(len(Vsav_pred[0, :, 0]))

    # Create a meshgrid from x and y values
    X, Y = np.meshgrid(x, y)

    # Plot the array as a color gradient
    plt.imshow(Vsav_pred_data_at_time_frame, cmap='jet', origin='lower', vmin=v_pred_min, vmax=v_pred_max)
    cbar = plt.colorbar()
    cbar.set_label('V')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Predition at {time_frame/100}s')
    plt.savefig(fig_name+'_Snapshot_2.tiff', format='tiff', dpi=500)
    plt.close()


def dice_score_calculator(V, V_pred, idx_data_larger):
    #Obtain the testing data
    testing_V = V[idx_data_larger]
    reshaped_V_pred = V_pred.reshape(-1, 1)
    testing_V_pred = reshaped_V_pred[idx_data_larger]

    #Apply mask for testing_V
    testing_V[testing_V > 0.5] = 1
    testing_V[testing_V < 0.5] = 0

    #Apply mask for testing_V_pred
    testing_V_pred[testing_V_pred > 0.5] = 1
    testing_V_pred[testing_V_pred < 0.5] = 0
    
    print(testing_V)
    print(testing_V.shape)
    print("____________________________________________")
    print(testing_V_pred)
    print(testing_V_pred.shape)
    
    #Find intersection
    intersection = np.sum(testing_V == testing_V_pred)

    print(intersection)

    #Find total area
    total_area = testing_V.shape[0] + testing_V_pred.shape[0]
    
    #Dice Score
    DICE_score = intersection*2/total_area

    print("---------------------------------------")
    print(f'Dice similarity score: {DICE_score}')
    print("---------------------------------------")


def  dice_score_calculator_1(Vsav, Vsav_pred, idx_data_larger):
    #Obtain the testing data
    dice_score_frame = []
    #As Vsav_pred is shaped as (x, y, y)
    len_time_index = Vsav.shape[2]
    for i in range(0, len_time_index-1):
        Vsav_frame = Vsav[:,:,i]
        Vsav_pred_frame = Vsav_pred[:,:,i]

        #Apply mask for Vsav_frame
        Vsav_frame[Vsav_frame > 0.5] = 1
        Vsav_frame[Vsav_frame <= 0.5] = 0

        #Apply mask for Vsav_pred_frame
        Vsav_pred_frame[Vsav_pred_frame > 0.5] = 1
        Vsav_pred_frame[Vsav_pred_frame <= 0.5] = 0

        #Find intersection
        reshaped_Vsav = Vsav_frame.reshape(-1,1)
        reshaped_Vsav_pred = Vsav_pred_frame.reshape(-1,1)

        print(reshaped_Vsav)
        print(reshaped_Vsav_pred)


        intersection = np.sum(reshaped_Vsav == reshaped_Vsav_pred)

        #Find total area
        Total_area = reshaped_Vsav.shape[0]+reshaped_Vsav_pred.shape[0]

        #Dice Score
        DICE_score = intersection*2/Total_area

        #Put into the dice_score_frame list
        dice_score_frame.append(DICE_score)

    #Find the average of DICE across all time frame
    DICE_sum = sum(dice_score_frame)
    average_DICE = DICE_sum / len(dice_score_frame)

    print("---------------------------------------")
    print(f'Dice similarity score: {average_DICE}')
    print("---------------------------------------")
