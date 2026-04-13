
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits 
import argparse


def get_DM_command_in_2D(cmd,Nx_act=12):
    # function so we can easily plot the DM shape (since DM grid is not perfectly square raw cmds can not be plotted in 2D immediately )
    #puts nan values in cmd positions that don't correspond to actuator on a square grid until cmd length is square number (12x12 for BMC multi-2.5 DM) so can be reshaped to 2D array to see what the command looks like on the DM.
    corner_indices = [0, Nx_act-1, Nx_act * (Nx_act-1), Nx_act*Nx_act]
    cmd_in_2D = list(cmd.copy())
    for i in corner_indices:
        cmd_in_2D.insert(i,np.nan)
    return( np.array(cmd_in_2D).reshape(Nx_act,Nx_act) )



def nice_heatmap_subplots( im_list , xlabel_list=None, ylabel_list=None, title_list=None, cbar_label_list=None, fontsize=15, cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None):

    n = len(im_list)
    fs = fontsize
    fig = plt.figure(figsize=(5*n, 5))

    for a in range(n) :
        ax1 = fig.add_subplot(int(f'1{n}{a+1}'))

        if vlims is not None:
            im1 = ax1.imshow(  im_list[a] , vmin = vlims[a][0], vmax = vlims[a][1])
        else:
            im1 = ax1.imshow(  im_list[a] )
        if title_list is not None:
            ax1.set_title( title_list[a] ,fontsize=fs)
        if xlabel_list is not None:
            ax1.set_xlabel( xlabel_list[a] ,fontsize=fs) 
        if ylabel_list is not None:
            ax1.set_ylabel( ylabel_list[a] ,fontsize=fs) 
        ax1.tick_params( labelsize=fs ) 

        if axis_off:
            ax1.axis('off')
        divider = make_axes_locatable(ax1)
        if cbar_orientation == 'bottom':
            cax = divider.append_axes('bottom', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
                
        elif cbar_orientation == 'top':
            cax = divider.append_axes('top', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='horizontal')
                
        else: # we put it on the right 
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar( im1, cax=cax, orientation='vertical')  
        
        if cbar_label_list is not None:
            cbar.set_label( cbar_label_list[a], rotation=0,fontsize=fs)
        cbar.ax.tick_params(labelsize=fs)
    if savefig is not None:
        plt.savefig( savefig , bbox_inches='tight', dpi=300) 



parser = argparse.ArgumentParser(description="Compare ZWFS setpoint intensities before and ufter update from RTC")


parser.add_argument(
    "--fname",
    type=str,
    help="path/to/file generated when we run the upadte command")
#--start_with_current_baldr_flat


args = parser.parse_args()

d = fits.open( args.fname )

#get_DM_command_in_2D will only work if the reconstructor was in dm space 
#im_list = [get_DM_command_in_2D(ii) for ii in [d['PRIOR_I0'].data, d['POST_I0'].data ]] + [get_DM_command_in_2D(d['PRIOR_I0'].data - d['POST_I0'].data)]
im_list = [get_DM_command_in_2D(ii[0]).shape(32,32) for ii in [d['PRIOR_I0'].data, d['POST_I0'].data ]] #+ [get_DM_command_in_2D(d['PRIOR_I0'].data - d['POST_I0'].data)]
#title_list = ['ZWFS INTENSITY SETPOINT\nPRIOR','ZWFS INTENSITY SETPOINT\nPOSTERIOR', r'$\Delta$'+'\nPRIOR - POSTERIOR']
title_list = ['ZWFS INTENSITY SETPOINT\nPRIOR','ZWFS INTENSITY SETPOINT\nPOSTERIOR']
nice_heatmap_subplots( im_list = im_list, 
                       title_list = title_list)

plt.show()

