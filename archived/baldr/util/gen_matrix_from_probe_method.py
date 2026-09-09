# baldr RTC has "run_probe_method" which has different methods to build interaction 
# matricies for example >run_probe_method ["I2A","/home/rtc/Downloads/test_i2a.fits"]
# will poke in/out sequence of 4 corners on DM which can be used to solve
# the DM actuator registration. This script is used to take the output fits
# file from the  run_probe_method for different methods and generate the 
# correct matrix from that method. 

import argparse
import json
import zmq
import matplotlib.pyplot as plt 
import numpy as np 
from astropy.io import fits 
import sys
sys.path.append("/home/rtc/Documents/asgard-alignment") # or whereever the asgard-alignment repository is

import common.DM_registration as DM_registration
from pyBaldr import utilities as util


# i should add a field to update toml file ! 
def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for RTC update script")

    # required
    parser.add_argument("method", type=str, help="Method to use")
    parser.add_argument("input_file", type=str, help="Input file path")

    # optional
    parser.add_argument("--output_file", type=str, default=None, help="Output file path (default: None)")
    parser.add_argument("--update_rtc", type=int, default=0, help="Update RTC flag (default: 0)")
    parser.add_argument("--savefig", type=str, default=None, help="directory where to save output plots. if savefig is None then they will not save (default)")

    return parser.parse_args()

def save_array_fits(arr, save_path, header_dict):
    hdr = fits.Header()
    for k, v in header_dict.items():
        hdr[k] = v
    fits.writeto(save_path, np.asarray(arr), header=hdr, overwrite=True)




args = parse_args()

if args.method == 'TT_OL_img':
    d = fits.open( args.input_file )
    tt_img = []
    for cc in np.arange(0,10,2): 
        c_in = np.median( d[1].data['DATA'][cc].reshape(d[1].header['N'],32,32),axis=0) 
        c_out = np.median( d[1].data['DATA'][cc+1].reshape(d[1].header['N'], 32,32),axis=0)
        delta_i = abs( c_in - c_out )
        tt_img.append( delta_i )

if args.method == 'I2A':
    
    d = fits.open( args.input_file )

    img_4_corners = [] # to hold the averaged differenced image from each in/out corner poke of DM
    for cc in np.arange(0,8,2):     
        c_in = np.median( d[1].data['DATA'][cc].reshape(d[1].header['N'],32,32),axis=0) 
        c_out = np.median( d[1].data['DATA'][cc+1].reshape(d[1].header['N'], 32,32),axis=0)
        delta_i = abs( c_in - c_out )
        # MANUAL pixel filter to set 
        #delta_i[delta_i > 40000] = np.mean( delta_i)
        img_4_corners.append(  delta_i )  

    dm_4_corners = DM_registration.get_inner_square_indices(outer_size=12, inner_offset=4) # flattened index of the DM actuator 


    transform_dict = DM_registration.calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners , debug=True, fig_path = args.savefig  )

    # From affine transform construct bilinear interpolation matrix on registered DM actuator positions
    #(image -> actuator transform) 
    img_tmp = img_4_corners[0].copy() # just copy tmp images to infer correct sizes

    x_target = np.array( [x for x,_ in transform_dict['actuator_coord_list_pixel_space']] )
    y_target = np.array( [y for _,y in transform_dict['actuator_coord_list_pixel_space']] )
    x_grid = np.arange(img_tmp.shape[0])
    y_grid = np.arange(img_tmp.shape[1])
    I2A = DM_registration.construct_bilinear_interpolation_matrix(image_shape=img_tmp.shape, 
                                            x_grid=x_grid, 
                                            y_grid=y_grid, 
                                            x_target=x_target,
                                            y_target=y_target)
    


    if args.output_file is not None:
        # put in fits file and save it to args.output_file
        header_dict = {"method":args.method, "source":args.input_file}
        save_array_fits(arr=I2A, save_path=args.output_file, header_dict=header_dict)

    if args.update_rtc: 
        # connect to Baldr RTC socket and update I2A via ZMQ
        addr = "tcp://127.0.0.1:6662"  # this will change depending on if we are in simulation mode
        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.REQ)
        s.RCVTIMEO = 5000  # ms
        s.SNDTIMEO = 5000  # ms
        s.connect(addr)

        # get the current config file 
        s.send_string('list_rtc_fields ""')
        rep = s.recv_json()

        # update the field (example!)
        #s.send_string('set_rtc_field "inj_signal.freq_hz",0.04')
        #rep = s.recv_json()
        s.send_string(f'set_rtc_field "matrices.I2A",{I2A}')
        rep = s.recv_json()

        print( "sucess?", rep['ok'] )
        if not rep['ok'] :
            print( '  error: ', rep['error'] )

        # close connection
        s.setsockopt(zmq.LINGER, 0)   # don't wait on unsent msgs
        s.close()  # closes the socket

if args.method == 'IM':

    # I need to always add camera gain and fps
    I0 = np.median(d[1].data['DATA'].reshape(280,d[1].header['N'],32,32),axis=(0,1)) 

    plt.figure(); plt.imshow(  I0 ); plt.show()
    

    img_4_corners = []  # also we can extract DM registration 
    corner_indicies = [50, 53, 86, 89]
    IM_2D = []

    for cc in range(140):
        c_in = np.median( d[1].data['DATA'][cc].reshape(d[1].header['N'],32,32),axis=0) 
        c_out = np.median( d[1].data['DATA'][cc+1].reshape(d[1].header['N'], 32,32),axis=0)

        IM_2D.append( c_out - c_in ) # normalize by gain and fps. Double check previous python scropt in asgard-alignment convention

        if cc in corner_indicies:
            img_4_corners.append( abs(c_out - c_in)  )





    IM = np.array( IM_2D ).reshape(140,-1)
    
    ## For DM registration which we get for free
    dm_4_corners = DM_registration.get_inner_square_indices(outer_size=12, inner_offset=4) # flattened index of the DM actuator 


    transform_dict = DM_registration.calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners , debug=True, fig_path = args.savefig  )

    # From affine transform construct bilinear interpolation matrix on registered DM actuator positions
    #(image -> actuator transform) 
    img_tmp = img_4_corners[0].copy() # just copy tmp images to infer correct sizes

    x_target = np.array( [x for x,_ in transform_dict['actuator_coord_list_pixel_space']] )
    y_target = np.array( [y for _,y in transform_dict['actuator_coord_list_pixel_space']] )
    x_grid = np.arange(img_tmp.shape[0])
    y_grid = np.arange(img_tmp.shape[1])
    I2A = DM_registration.construct_bilinear_interpolation_matrix(image_shape=img_tmp.shape, 
                                            x_grid=x_grid, 
                                            y_grid=y_grid, 
                                            x_target=x_target,
                                            y_target=y_target)
    
    if args.output_file is not None:
        # put in fits file and save it to args.output_file
        header_dict = {"method":args.method, "source":args.input_file}
        save_array_fits(arr=IM, save_path=args.output_file, header_dict=header_dict)

        # connect to Baldr RTC socket and update I2A via ZMQ
        addr = "tcp://127.0.0.1:6662"  # this will change depending on if we are in simulation mode
        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.REQ)
        s.RCVTIMEO = 5000  # ms
        s.SNDTIMEO = 5000  # ms
        s.connect(addr)

        # get the current config file 
        s.send_string('list_rtc_fields ""')
        rep = s.recv_json()

        # update the field
        s.send_string(f'set_rtc_field "matrices.IM",{IM}')
        rep = s.recv_json()

        # TO DO , update dependancies... Do I compute new control matricies? 


else:
    raise UserWarning(f"Method {args.method} not implemented yet. Check you input the correct method.")