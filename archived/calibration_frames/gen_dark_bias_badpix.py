import zmq
from astropy.io import fits
import argparse
import time
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt 
import sys
sys.path.insert(0, '/home/asg/Progs/repos/asgard-alignment/asgard_alignment/')
from asgard_alignment import FLI_Cameras as FLI # depends on xao shm

########################################################################
# standardized format to generate bias, dark and bad pixel maps 
# for Baldr/Heim
########################################################################

tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
tstamp_rough = datetime.datetime.now().strftime("%d-%m-%Y")

parser = argparse.ArgumentParser(description="generate darks for different gain settings on CRED ONE")

parser.add_argument(
    '--data_path',
    type=str,
    default=f"/home/asg//Progs/repos/dcs/calibration_frames/products/{tstamp_rough}/",
    help="Path to the directory for storing dark data. Default: %(default)s"
)

parser.add_argument(
    '--mode',
    type=str,
    default="globalresetcds",
    help="C-RED ONE mode Try globalresetbursts or globalresetcds. Default: %(default)s"
)
parser.add_argument(
    '--gains',
    type=int,
    nargs='+',
    default=[1], # , 5, 10, 20],
    help="List of gains to apply when collecting dark frames. Default: %(default)s"
)
parser.add_argument(
    '--fps',
    type=float,
    default=1000,
    help="frame rate for darks. Default: %(default)s"
)

parser.add_argument(
    '--method',
    type=str,
    default="traditional",
    help="What method to generate darks? traditional, linear_fit,log-quadratic_fit. Default: %(default)s"
)

parser.add_argument(
    '--mean_threshold',
    type=float,
    default=4.0,
    help="mean threshold for calculating bad pixels on darks. Default: %(default)s"
)
parser.add_argument(
    '--std_threshold',
    type=float,
    default=10.0,
    help="std threshold for calculating bad pixels on darks. Default: %(default)s"
)
parser.add_argument("--number_of_frames", 
                    type=int, 
                    default=1000, 
                    help="number of frames to take for dark")

parser.add_argument("--host", type=str, default="172.16.8.6", help="Server host")
parser.add_argument("--port", type=int, default=5555, help="Server port")
parser.add_argument(
    "--timeout", type=int, default=2000, help="Response timeout in milliseconds"
)



args = parser.parse_args()

context = zmq.Context()
context.socket(zmq.REQ)
mds_socket = context.socket(zmq.REQ)
mds_socket.setsockopt(zmq.RCVTIMEO, args.timeout)
mds_socket.connect( f"tcp://{args.host}:{args.port}")

sleeptime = 10 # seconds <--- required to get ride of persistance of pupils 

# to store master darks 
if not os.path.exists( args.data_path ):
    os.makedirs( args.data_path )

# to store raw darks
for red_pth in ["RAW_DARKS/","RAW_BIAS/","MASTER_BIAS/","MASTER_DARK/","BAD_PIXEL_MAP/"] :
    if not os.path.exists( args.data_path + red_pth  ):
        os.makedirs( args.data_path + red_pth  )
        print(f"made directory {args.data_path + red_pth}")



#cc =  shm("/dev/shm/cred1.im.shm")  # testing 
c = FLI.fli(config_file_path = "/home/asg/Progs/repos/asgard-alignment/config_files/")

aduoffset = 1000 # hard coded for consistency 

# default aduoffset to avoid overflow of int on CRED 
c.send_fli_cmd(f"set aduoffset {aduoffset}")

time.sleep(1
           )
#if args.fps is not None: 
c.send_fli_cmd(f"set fps {args.fps}")

print("CHECK YOU ARE IN THE CORRECT CAMERA MODE: (SHOULD BE {args.mode}). I DON'T DO THIS FOR YOU NOW!")
#time.sleep(1) # Don't do this it can make things strange if done while server running! 
#c.send_fli_cmd(f"set mode {args.mode}")

time.sleep(1)

# camera config
config_tmp = c.config.copy()


# integration time for the CRED 1. 
dt = round( 1/float(config_tmp["fps"]) , 5 )

# try turn off source 
message = "off SBB"
mds_socket.send_string(message)
response = mds_socket.recv_string()#.decode("ascii")
print( response )

print(f'turning off source and waiting {sleeptime}s')
time.sleep(sleeptime) # wait a bit to settle

# I DECIDED AGAINST MULTIPLE EXTENSIONS IN A FITS HDULIST
# -- INSTEAD KEEP FILES LIGHTWEIGHT AND QUERYABLE FROM FILE NAME
# master_bias_hdulist = fits.HDUList([] ) 
# master_dark_hdulist = fits.HDUList([] ) 
# master_badpixel_hdulist = fits.HDUList([] ) 
# master_convgain_hdulist = fits.HDUList([] ) 


print('...getting frames')

for gain in args.gains: #np.arange(1,args.max_gain):

    print(f"   gain = {gain}")

    c.send_fli_cmd(f"set gain {gain}")

    # edit our cponfig dict without bothering to comunicate with camera
    config_tmp["gain"] = gain 

    resp = c.send_fli_cmd( "maxfps" )
    maxfps = float( resp.split(": ")[-1].split("\\r")[0] )  # -1 

    # ----------BIAS 
    c.send_fli_cmd( f"set fps {maxfps}")

    time.sleep(5)

    bias_list = c.mySHM.get_latest_data() #c.get_some_frames(number_of_frames = args.number_of_frames, apply_manual_reduction=False, timeout_limit = 20000 )

    # return to previous fps
    c.send_fli_cmd(f"set fps {config_tmp['fps']}")

    if args.method == "traditional": # bias is fastest frame rate and we get a dark per fps

        master_bias = np.mean(bias_list, axis=0) # [adu]

        time.sleep(3) 

        # ----------DARK
        dark_list = c.mySHM.get_latest_data() #c.get_some_frames(number_of_frames = args.number_of_frames, apply_manual_reduction=False, timeout_limit = 20000 )

        #if  "globalresetcds" in args.mode:
        master_dark = np.clip( np.mean(dark_list, axis=0)  -  master_bias, 0, np.inf) / dt #[adu/s]

        #elif "globalresetcds" not in args.mode: # we don't subtract off bias ! 
        #    master_dark =  np.mean(dark_list, axis=0)   / dt #[adu/s]


    ### This should be the standard mode 
    elif args.method == "linear_fit": # we fit a pixel wise linear fit around the gieven integration time for ADU/s

        
        frames = {}
        # fit linear model ADU/s around the fps. We find reasonable linear range (not background limited) in 100-400Hz , beyond this reset anaonomly dominates
        for fps in np.linspace( np.max([ 50, float(config_tmp['fps'])-100])  , float(config_tmp['fps'])+100 , 5): # Trye not go too low to get background, but keep to moderate FPS to avoid being dominated by reset anomaly 
            c.send_fli_cmd(f"set fps {fps}")
            time.sleep(10)

            frames[fps] = np.mean( c.mySHM.get_latest_data() ,axis=0) 


        fps_values = list(frames.keys())  # e.g. [10, 20, 30, 40]

        # inverse fps to get the integration time 
        x = 1/np.array(fps_values, dtype=np.float64)  # independent variable (fps)
        n_fps = len(x)

        # Stack images: result shape will be (n_fps, height, width)
        images = np.stack([frames[fps] for fps in fps_values], axis=0)

        # Define a helper function that returns the coefficients using np.polyfit.
        # np.polyfit returns coefficients ordered [slope, intercept] for a degree-1 polynomial.
        def polyfit_pixel(y_vals):
            return np.polyfit(x, y_vals, 1)  # y_vals should be shape (n_fps,)

        # Apply this function along the fps axis (axis 0) for every pixel.
        # This will result in an array of shape (2, height, width), where:
        #   - [0, :, :] is the slope (m)
        #   - [1, :, :] is the intercept (b)
        coefs = np.apply_along_axis(polyfit_pixel, 0, images)

        # Extract slopes and intercepts as 2D arrays.
        master_dark  = coefs[0]  # ADU/s for each pixel, shape (height, width)
        master_bias  = coefs[1]  # intercept (bias + aduoffset) for each pixel, shape (height, width)


        # return to previous fps
        c.send_fli_cmd(f"set fps {config_tmp['fps']}")

        time.sleep(3) 

        dark_list = c.mySHM.get_latest_data() #c.get_some_frames(number_of_frames = args.number_of_frames, apply_manual_reduction=False, timeout_limit = 20000 )

        # ------------------------------
        # Plot a few pixel fits as subplots
        # ------------------------------
        # height, width = images.shape[1], images.shape[2]
        # sample_pixels = [
        #     (100, 100),
        #     (height // 2, width // 2),
        #     (height - 1, width - 1),
        #     (height // 4, width // 4)
        # ]

        # # Set up subplots (one subplot per chosen pixel)
        # fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # axes = axes.flatten()

        # for ax, (i, j) in zip(axes, sample_pixels):
        #     y_data = images[:, i, j]
            
        #     # Compute the fitted line: y = m*x + b for pixel (i,j)
        #     fitted_line = master_dark[i, j] * x + master_bias[i, j]

        #     ax.scatter(x, y_data, color='blue', label='Data')
        #     ax.plot(x, fitted_line, color='red', label='Fit')
        #     ax.set_xscale('log')
        
        #     ax.set_title(f'Pixel ({i},{j})\nm: {master_dark[i, j]:.4e}, b: {master_bias[i, j]:.4e}')
        #     ax.set_xlabel('Integration time [seconds]')
        #     ax.set_ylabel('Pixel Intensity')
        #     ax.legend()

        # plt.tight_layout()
        #plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')

    elif args.method == "log-quadratic_fit": # This seemss to fit the CDS mode very well! defines boundary of dark - background

        frames = {}
        for fps in [50, 100, 300, 600, 1000]: #got to higher FPS to capture and model non-linearities:
            c.send_fli_cmd(f"set fps {fps}")
            time.sleep(10)

            frames[fps] = np.mean( c.mySHM.get_latest_data() ,axis=0) 


        fps_values = list(frames.keys())  # e.g. [10, 20, 30, 40]

        # inverse fps to get the integration time 
        x = 1/np.array(fps_values, dtype=np.float64)  # independent variable (fps)
        n_fps = len(x)

        # Stack images: result shape will be (n_fps, height, width)
        images = np.stack([frames[fps] for fps in fps_values], axis=0)

        # For the quadratic fit, use the natural logarithm of x:
        log_x = np.log(x)
        # Define a helper function that fits y = a*(log(x))^2 + b*log(x) + c for a given pixel vector
        def polyfit_pixel(y_vals):
            # y_vals is a 1D array of length n_fps; we use log_x (computed above) as the independent variable.
            # np.polyfit returns coefficients [a, b, c] such that y = a*log(x)^2 + b*log(x) + c.
            return np.polyfit(log_x, y_vals, 2)

        # Apply the function along axis 0 (across the integration time dimension) for each pixel.
        # Resulting coefs has shape (3, height, width):
        #   coefs[0] = a (quadratic coefficient)
        #   coefs[1] = b (linear coefficient)
        #   coefs[2] = c (constant/intercept)
        coefs = np.apply_along_axis(polyfit_pixel, 0, images)

        # Extract coefficient arrays
        a_coef = coefs[0]  # Quadratic coefficient, 2D array (height, width)
        b_coef = coefs[1]  # Linear coefficient, 2D array (height, width)
        c_coef = coefs[2]  # Constant term, 2D array (height, width)

        master_bias = c_coef # This is more of an estimate of reset anomaly in CDS mode - these models are better use to find transition regions
        master_dark = b_coef # linearized 

        # return to previous fps
        c.send_fli_cmd(f"set fps {config_tmp['fps']}")

        time.sleep(3) 
        
        dark_list = c.mySHM.get_latest_data() #c.get_some_frames(number_of_frames = args.number_of_frames, apply_manual_reduction=False, timeout_limit = 20000 )


        # ------------------------------
        # Plot a few pixel fits as subplots
        # ------------------------------

        # # Choose several sample pixel coordinates.
        # height, width = images.shape[1], images.shape[2]
        # sample_pixels = [
        #     (100, 100),
        #     (height // 2, width // 2),
        #     (height - 1, width - 1),
        #     (height // 4, width // 4)
        # ]

        # # Set up subplots (e.g. 2x2 grid)
        # fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        # axes = axes.flatten()

        # for ax, (i, j) in zip(axes, sample_pixels):
        #     # Extract the pixel's intensity data across integration times
        #     y_data = images[:, i, j]

        #     # Compute the fitted values using the log-quadratic model:
        #     # fitted_line = a*(log(x))^2 + b*log(x) + c
        #     fitted_line = a_coef[i, j] * (log_x ** 2) + b_coef[i, j] * log_x + c_coef[i, j]

        #     # Plot the original data and the fitted line:
        #     ax.scatter(x, y_data, color='blue', label='Data')
        #     ax.plot(x, fitted_line, color='red', label='Fit')
            
        #     # Use a logarithmic x-axis (integration times)
        #     ax.set_xscale('log')
            
        #     ax.set_title(f'Pixel ({i},{j})\n'
        #                 f'a: {a_coef[i, j]:.4e}, b: {b_coef[i, j]:.4e}, c: {c_coef[i, j]:.4e}')
        #     ax.set_xlabel('Integration time [s]',fontsize=15)
        #     ax.set_ylabel('Pixel Intensity [ADU]',fontsize=15)
        #     ax.legend()

        # plt.tight_layout()
        # plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')
        # plt.show()

    # ----------writing raw dakrs
    dark_hdu = fits.PrimaryHDU(dark_list)
    dark_hdu.header['EXTNAME'] = 'DARK_FRAMES'  
    dark_hdu.header['UNITS'] = "ADU"
    dark_hdu.header['DATE'] = tstamp
    for k, v in config_tmp.items():
        dark_hdu.header[k] = v


    hdulist = fits.HDUList([dark_hdu]) #, badpix_hdu, conv_gain_hdu])

    fname_raw = args.data_path + f"RAW_DARKS/raw_darks_{args.mode}_fps-{config_tmp['fps']}_gain-{config_tmp['gain']}_{tstamp}.fits"
    hdulist.writeto(fname_raw, overwrite=True)
    print( f"   wrote raw darks for gain {gain}:\n    {fname_raw}" )

    # ----------Estimate pixelwise conversion gain (e-/ADU)
    # Assumes the sinal is dominated by shot noise
	# The system is linear
	# The bias (offset) has been properly subtracted
    # THIS NEEDS TO BE DONE ON INTERNAL FLAT 
    # dark_minus_bias = np.array( dark_list ) - np.mean(bias_list, axis=0)
    # mean_dark = np.mean( dark_minus_bias, axis=0)
    # var_dark = np.var( dark_minus_bias, axis=0) 
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     conversion_gain = np.where(var_dark > 0, mean_dark / var_dark, 0)

    # conv_gain_hdu = fits.ImageHDU(conversion_gain)
    # conv_gain_hdu.header['EXTNAME'] = f'CONV_GAIN_GAIN-{gain}'
    # conv_gain_hdu.header['UNITS'] = 'e-/ADU'
    # for k, v in config_tmp.items():
    #     conv_gain_hdu.header[k] = v    
    # conv_gain_hdu.header['DATE'] = tstamp

    # ---------- Calculate bad_pixel_map 
    _, bad_pixel_mask = FLI.get_bad_pixels( dark_list, std_threshold = args.std_threshold, mean_threshold = args.mean_threshold)
    badpix_hdu = fits.PrimaryHDU( np.array( bad_pixel_mask).astype(int) )
    badpix_hdu.header['EXTNAME'] = f'BAD_PIXEL_MAP_GAIN-{gain}'  
    for k, v in config_tmp.items():
        badpix_hdu.header[k] = v
    badpix_hdu.header['STD_THR'] = args.std_threshold
    badpix_hdu.header['MEAN_THR'] = args.mean_threshold
    badpix_hdu.header['DATE'] = tstamp


    fname_badpix_master = args.data_path +  f"BAD_PIXEL_MAP/master_bad_pixel_map_{args.mode}_fps-{config_tmp['fps']}_gain-{config_tmp['gain']}_{tstamp}.fits"
    badpix_hdu.writeto(fname_badpix_master, overwrite=True)
    print( f"   wrote bad pixel mask:\n    {fname_badpix_master}" )


    # ----------writing raw bais
    bias_hdu = fits.PrimaryHDU(bias_list)
    bias_hdu.header['EXTNAME'] = f'BIAS_GAIN-{gain}'  
    bias_hdu.header['UNITS'] = "ADU"
    bias_hdu.header['DATE'] = tstamp
    bias_hdu.header['METHOD'] = f"{args.method}"
    for k, v in config_tmp.items():
        bias_hdu.header[k] = v

    hdulist = fits.HDUList([bias_hdu])

    fname_raw = args.data_path +  f"RAW_BIAS/raw_bias_{args.mode}_maxfps-{maxfps}_gain-{config_tmp['gain']}_{tstamp}.fits"
    hdulist.writeto(fname_raw, overwrite=True)
    print( f"   wrote raw bias for gain {gain}:\n    {fname_raw}" )

    # ----------writing MASTER DARK
    master_dark_hdu = fits.PrimaryHDU(master_dark)
    master_dark_hdu.header['EXTNAME'] = f"DARK_GAIN-{gain}"
    master_dark_hdu.header['UNITS'] = "ADU/s"
    master_dark_hdu.header['DATE'] = tstamp
    master_dark_hdu.header['SYSTEM'] = "BALDR-HEIM_CRED1"
    master_dark_hdu.header['METHOD'] = f"{args.method}"
    for k, v in config_tmp.items():
        master_dark_hdu.header[k] = v

    # print("calculated master dark and appending the fits")
    # master_dark_hdulist.append( master_dark_hdu )
    fname_dark_master = args.data_path + f"MASTER_DARK/master_darks_{args.mode}_fps-{config_tmp['fps']}_gain-{config_tmp['gain']}_{tstamp}.fits"
    master_dark_hdu.writeto(fname_dark_master, overwrite=True)
    print( f"   wrote master darks:\n    {fname_dark_master}" )

    # ----------writing MASTER BIAS
    master_bias_hdu = fits.PrimaryHDU(master_bias)
    master_bias_hdu.header['EXTNAME'] = f"BIAS_GAIN-{gain}"
    master_bias_hdu.header['UNITS'] = "ADU"
    master_bias_hdu.header['DATE'] = tstamp
    master_bias_hdu.header['SYSTEM'] = "BALDR-HEIM_CRED1"
    for k, v in config_tmp.items():
        master_bias_hdu.header[k] = v


    fname_bias_master = args.data_path + f"MASTER_BIAS/master_bias_{args.mode}_fps-{config_tmp['fps']}_gain-{config_tmp['gain']}_{tstamp}.fits"
    master_bias_hdu.writeto(fname_bias_master, overwrite=True)
    print( f"   wrote master bias:\n    {fname_bias_master}" )


    # ----------writing FITTED COEFFICIENTS IF A HIGHER ORDER MODEL 

    if args.method == "log-quadratic_fit":

        if not os.path.exists( args.data_path + 'DARK_LOG-QUAD_FIT'  ):
            os.makedirs( args.data_path + 'DARK_LOG-QUAD_FIT'   )
            print(f"made directory {args.data_path + 'DARK_LOG-QUAD_FIT' }")
        model_coe = fits.PrimaryHDU( [a_coef, b_coef, c_coef] )
        model_coe.header['EXTNAME'] = f"FITTED_MODEL"
        model_coe.header['DATE'] = tstamp
        model_coe.header['SYSTEM'] = "BALDR-HEIM_CRED1"
        model_coe.header['METHOD'] = f"{args.method}"

        for k, v in config_tmp.items():
            model_coe.header[k] = v

        # print("calculated master dark and appending the fits")
        # master_dark_hdulist.append( master_dark_hdu )
        fname_model_coe = args.data_path +  f"DARK_LOG-QUAD_FIT/dark_log-quadratic_fit_{args.mode}_fps-{config_tmp['fps']}_gain-{config_tmp['gain']}_{tstamp}.fits"
        model_coe.writeto(fname_model_coe, overwrite=True)
        print( f"   wrote master darks:\n    {fname_model_coe}" )

    #print("calculated master bias and appending the fits")
    #master_bias_hdulist.append( master_bias_hdu )

    # ---------- MASTER BAD PIXEL MAP
    #print("calculated master bad pixel map and appending the fits")
    #master_badpixel_hdulist.append( badpix_hdu )

    # ---------- CONVERSION GAIN MAP
    #print("calculated master conversion gain map and appending the fits")
    #master_convgain_hdulist.append( conv_gain_hdu )

# fname_dark_master = args.data_path + f"master_darks_adu_p_sec_fps-{config_tmp['fps']}_{tstamp}.fits"
# master_dark_hdulist.writeto(fname_dark_master, overwrite=True)
# print( f"   wrote master darks:\n    {fname_dark_master}" )

# fname_bias_master = args.data_path + f"master_bias_fps-{config_tmp['fps']}_{tstamp}.fits"
# master_bias_hdulist.writeto(fname_bias_master, overwrite=True)
# print( f"   wrote master bias:\n    {fname_bias_master}" )


# fname_badpix_master = args.data_path + f"master_bad_pixel_map_fps-{config_tmp['fps']}_{tstamp}.fits"
# master_badpixel_hdulist.writeto(fname_badpix_master, overwrite=True)
# print( f"   wrote bad pixel mask:\n    {fname_badpix_master}" )

# This gets done on internal flat field 
# fname_convgain_master = args.data_path + f"conversion_gain_map_fps-{config_tmp['fps']}_{tstamp}.fits"
# master_convgain_hdulist.writeto(fname_convgain_master, overwrite=True)
# print( f"   wrote conversion gain:\n    {fname_convgain_master}" )



# ---------- TEST SECTION ----------
verify_res = False 
if verify_res: # 


    # plt.figure(); plt.hist(  (~bad_pixel_mask * b).reshape(-1), bins=np.linspace(aduoffset-200, aduoffset+200) );plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')
 
    # plt.figure(); plt.hist(  (~bad_pixel_mask * m).reshape(-1) , bins= list( -np.logspace(0,3,100)[::-1] ) + list( np.logspace(0,3,100) )) #bins=np.logspace(-4,0,100)); 
    # plt.xscale( 'symlog',linthresh=1e3 ) #'symlog',linthresh=1e-2)
    # plt.xlabel("Pixel-wise Slope (ADU/s) - low fps gain = 10")
    # plt.ylabel("Frequency")
    # plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')


    # plt.figure(); plt.imshow( np.log10( 1/(~bad_pixel_mask * np.exp( -b_coef / (2*a_coef) ) ) ), vmin = 1, vmax=3 ); plt.colorbar(label="log10 FPS Vertex") 
    # plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')

    # #, vmin = -1e-2, vmax=1e-2
    # plt.figure(); plt.imshow( ~bad_pixel_mask * m  , vmin = -150, vmax=150); plt.colorbar(label="ADU/s") 
    # plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')

    # plt.figure(); plt.imshow(  ~bad_pixel_mask * master_bias ); plt.colorbar() 
    # plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')

    # plt.figure(); plt.imshow(  ~bad_pixel_mask * np.mean(dark_list,axis=0) ); plt.colorbar() 
    # plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')

    # plt.figure(); plt.imshow(  ~bad_pixel_mask * (np.mean(dark_list,axis=0) -  master_bias) ); plt.colorbar() 
    # plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')

    # plt.figure(); plt.imshow(  ~bad_pixel_mask * master_dark ); plt.colorbar() 
    # plt.savefig('/home/asg/Progs/repos/asgard-alignment/delme.png')


    print("\n--- Running validation test at gain = mid range ---")

    gain = np.max([1, args.gains[len(args.gains)//2]])
    # Set gain to 
    c.send_fli_cmd(f"set gain {gain}")

    # Get camera config and fps
    fps = float(config_tmp["fps"])   
    dt = 1.0 / fps

    # Acquire test dark
    test_dark = np.mean(c.get_some_frames(number_of_frames=100, apply_manual_reduction=False), axis=0)  # [ADU]

    # Load corresponding master bias and master dark
    #master_bias_test = master_bias_hdulist[f"BIAS_GAIN-{gain}"].data   # gain = 5 → index = 4 (0-based)
    #master_dark_test = master_dark_hdulist[f"DARK_GAIN-{gain}"].data   # [ADU/s]
    #bad_pixel_mask = master_badpixel_hdulist[f"BAD_PIXEL_MAP_GAIN-{gain}"].data

    master_bias_test_fits = fits.open( args.data_path + f"MASTER_BIAS/master_bias_fps-{config_tmp['fps']}_gain-{gain}_{tstamp}.fits" )
    master_dark_test_fits = fits.open(args.data_path + f"MASTER_DARK/master_darks_adu_p_sec_fps-{config_tmp['fps']}_gain-{gain}_{tstamp}.fits" )
    bad_pixel_mask_fits = fits.open(args.data_path + f"BAD_PIXEL_MAP/master_bad_pixel_map_fps-{config_tmp['fps']}_gain-{gain}_{tstamp}.fits" )


    master_bias_test = master_bias_test_fits[0].data
    master_dark_test = master_dark_test_fits[0].data
    bad_pixel_mask = bad_pixel_mask_fits[0].data

    # Subtract bias
    test_dark_minus_bias = test_dark - master_bias_test  # [ADU]

    # Convert master dark to [ADU] by multiplying with dt
    expected_dark = master_dark_test * dt  # [ADU]

    # Compute pixelwise error and statistics
    error_map = test_dark_minus_bias - expected_dark # ADU]
    mean_error = np.mean(error_map[~bad_pixel_mask ])
    std_error = np.std( error_map[~bad_pixel_mask ])

    plt.figure(); plt.imshow( ~bad_pixel_mask * error_map) ;plt.colorbar() ;plt.savefig('delme.png')

    print(f"Validation test for gain  = {gain}:")
    print(f" - FPS = {fps} -> dt = {dt:.5f} s")
    print(f" - Mean pixel error [ADU] = {mean_error:.3f}")
    print(f" - Std  pixel error [ADU] = {std_error:.3f}")


# try turn source back on 
message = "on SBB"
mds_socket.send_string(message)
response = mds_socket.recv_string()#.decode("ascii")
print( response )
time.sleep(2)

# close camera SHM and set gain = 1 
c.close(erase_file=False)


print(f"DONE.")

