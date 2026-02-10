# To auto align the baldr pupils when on sky
# ===========================================
# this reads the active rtc config toml file which contains 
# the latest pupil center calibration in pixel space
# it then acquires an image (does not check the system state! 
# This is up to the users) and fits an ellipse to it and measures 
# the center it then publishes offsets (in mm) to the MCS so WAG can apply 
# an offset to the VLTI VCM (or whatever device will move the pupil)
# to align to specified reference pixel. 

# Its essential you use the active (rtc) toml config file. This should
# be done automatically with no input, but you can optionally specify a file


# HERE WE ONLY EVER UPDATE ONE BEAM AT A TIME!   This is because of how wag only accepts ranges of beams to update and not individuals


from xaosim.shmlib import shm
import numpy as np
import matplotlib.pyplot as plt
import time
import zmq
import os
import argparse
import json
import toml
from scipy.ndimage import gaussian_filter
from scipy.optimize import leastsq
import dcs.ZMQutils




class BaldrAA:

    def __init__(self,  beam, mode='bright', config_filepath=None, output="mcs", savepath = None ) :

        self.beam = beam

        self.mode = mode

        if mode.strip().lower() not in ['bright','faint']:
            raise UserWarning( "incorrect input mode. It must either be'bright' or 'faint'")
        
        self.filepath = config_filepath      

        # if output not in ["mcs","internal"]:
        #     raise ValueError("Output must be 'internal' or 'mcs'")

        # for now dont do mds 
        #self.mds = self._open_mds_connection()

        self.stream = self._open_stream_connection(self.beam)

        if "mcs" not in output:
            raise UserWarning("this is only implemented to update mcs or test mcs. Not internal alignment. This is done in asgard-algnment/calibration where we deal with phasemasks. Later I could put in here but will add dependancies / add complexity. Lets keep this critical onsky function simple ;) ")
        else:
            self.output = output

        if self.output.strip().lower() == "mcs":
            self.mcs_client = dcs.ZMQutils.ZmqReq("tcp://192.168.100.2:7019")

        self.savepath = savepath

    # # MDS interface - if one day i want it here
    # def _open_mds_connection(self):
    #     context = zmq.Context()
    #     socket = context.socket(zmq.REQ)
    #     socket.setsockopt(zmq.RCVTIMEO, 10000)
    #     server_address = "tcp://192.168.100.2:5555"
    #     socket.connect(server_address)
    #     return socket
    
    def _open_stream_connection(self, beam):
        stream_path = f"/dev/shm/baldr{beam}.im.shm"#"/dev/shm/cred1.im.shm"
        if not os.path.exists(stream_path):
            raise FileNotFoundError(f"Stream file {stream_path} does not exist.")
        return shm(stream_path)

    def get_frame(self):
        full_frame = self.stream.get_data()#.mean(0)
        return full_frame

    def _get_active_rtc_config_toml(self):
        if self.filepath is None: # automatic loading
            # by convention there is always an active toml file
            # at this default path with baldr rtc config
            # when we are ready, we load this to the active 
            # rtc path by asgard-alignment/calibration/copy_configs_to_stable_versions.sh
            # it is EXTREMELY unlikely that this should have different pupil registrations from the active RTC 
            # This would only happen if someone loaded a different config file to the rtc
            # for now we opt for the option that is least likely to break something 
            # and require tired troubleshooting 
            default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml") 
            
            return( default_toml.replace( '#',str(self.beam) ) )
        
        else: # you already know what toml file you want to use
            return(self.filepath)     
            
        
    def extract_pupil_references(self):

        found_file = self._get_active_rtc_config_toml(  )

        with open( found_file ) as file:
            config_dict = toml.load(file)

            # for references of where this information is generated in 
            # the config files scripts : 
            # - asgard-alignment/calibration/detect_cropped_pupils_coords.py
            # - asgard-alignment/calibration/pupil_registration.py

            # the baldr subframes
            all_subframes = config_dict.get("baldr_pupils", None) # { beam1:[r1,r2,c1,c2], beam2:[r1,r2,c1,c2],... }
            if all_subframes is None:
                # how should we handle failure ?
                raise UserWarning( f"'baldr_pupils' entry doesn't exist in provided toml file : {self.filepath}")

            # the Baldr configured subframe for this beam
            subframe = all_subframes[str(self.beam)]
            
            ref_x = config_dict.get( f"beam{self.beam}", {}).get( "pupil_ellipse_fit",{}).get("center_x",None)
            if ref_x is None:
                # how should we handle failure ? 
                raise UserWarning( f"'beam{self.beam}.pupil_ellipse_fit' entry doesn't exist in provided toml file : {self.filepath}")


            ref_y = config_dict.get( f"beam{self.beam}", {}).get( "pupil_ellipse_fit",{}).get("center_y",None)
            if ref_y is None:
                # how should we handle failure ?
                raise UserWarning( f"'beam{self.beam}.pupil_ellipse_fit' entry doesn't exist in provided toml file : {self.filepath}")

        return( ref_x, ref_y, subframe )



    def detect_pupil(self, image, sigma=2, threshold=0.5, plot=False, savepath=None):
        """
        Detects an elliptical pupil (with possible rotation) in a cropped image using edge detection 
        and least-squares fitting. Returns both the ellipse parameters and a pupil mask.

        The ellipse is modeled by:

            ((x - cx)*cos(theta) + (y - cy)*sin(theta))^2 / a^2 +
            (-(x - cx)*sin(theta) + (y - cy)*cos(theta))^2 / b^2 = 1

        Parameters:
            image (2D array): Cropped grayscale image containing a single pupil.
            sigma (float): Standard deviation for Gaussian smoothing.
            threshold (float): Threshold factor for edge detection.
            plot (bool): If True, displays the image with the fitted ellipse overlay.
            savepath (str): If provided, the plot is saved to this path.

        Returns:
            (center_x, center_y, a, b, theta, pupil_mask)
            where (center_x, center_y) is the ellipse center,
                    a and b are the semimajor and semiminor axes,
                    theta is the rotation angle in radians,
                    pupil_mask is a 2D boolean array (True = inside ellipse).
        """
        # Normalize the image
        if image.max() != 0:

            image = image / image.max()

        image[ image > np.quantile(image, 0.98) ] = np.median( image )

        # Smooth the image
        smoothed_image = gaussian_filter(image, sigma=sigma)
        
        # Compute gradients (Sobel-like edge detection)
        grad_x = np.gradient(smoothed_image, axis=1)
        grad_y = np.gradient(smoothed_image, axis=0)
        edges = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold edges to create a binary mask
        binary_edges = edges > (threshold * edges.max())
        
        # Get edge pixel coordinates
        y_coords, x_coords = np.nonzero(binary_edges)
        
        # Initial guess: center from mean, radius from average distance, and theta = 0.
        def initial_guess(x, y):
            center_x = np.mean(x)
            center_y = np.mean(y)
            r_init = np.sqrt(np.mean((x - center_x)**2 + (y - center_y)**2))
            return center_x, center_y, r_init, r_init, 0.0  # (cx, cy, a, b, theta)
        
        # Ellipse model function with rotation.
        def ellipse_model(params, x, y):
            cx, cy, a, b, theta = params
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            x_shift = x - cx
            y_shift = y - cy
            xp =  cos_t * x_shift + sin_t * y_shift
            yp = -sin_t * x_shift + cos_t * y_shift
            # Model: xp^2/a^2 + yp^2/b^2 = 1 => residual = sqrt(...) - 1
            return np.sqrt((xp/a)**2 + (yp/b)**2) - 1.0

        # Fit via least squares.
        guess = initial_guess(x_coords, y_coords)
        result, _ = leastsq(ellipse_model, guess, args=(x_coords, y_coords))
        center_x, center_y, a, b, theta = result
        
        # Create a boolean pupil mask for the fitted ellipse
        yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_shift = xx - center_x
        y_shift = yy - center_y
        xp = cos_t * x_shift + sin_t * y_shift
        yp = -sin_t * x_shift + cos_t * y_shift
        pupil_mask = (xp/a)**2 + (yp/b)**2 <= 1

        if savepath is not None:
            # Overlay for visualization
            overlay = np.zeros_like(image)
            overlay[pupil_mask] = 1
            
            plt.figure(figsize=(6, 6))
            plt.imshow(image, cmap="gray", origin="upper")
            plt.contour(binary_edges, colors="cyan", linewidths=1)
            plt.contour(overlay, colors="red", linewidths=1)
            plt.scatter(center_x, center_y, color="blue", marker="+")
            plt.title("Detected Pupil with Fitted Ellipse")
            #if savepath is not None:
            plt.savefig(savepath)
            #plt.show()
            plt.close()
        return center_x, center_y, a, b, theta, pupil_mask



    def measure_pupil_center( self ):
        # get the configured (registered) x,y pixels in the beam subframe 
        ref_x, ref_y, rc = self.extract_pupil_references(  ) # rc is the subframe pixels [r1,r2,c1,c2] in global frame coord

        sub_img_list = []
        for _ in range(50):
            sub_img_list.append( self.get_frame() )
            time.sleep(0.01)

        # simple removal of bad pixels etc 
        #r_sub_img_list = reduce(sub_img_list)
        sub_img = np.median(r_sub_img_list, axis=0)

        # full_img = self.get_frame() # average of 200 frames

        # # check if in cropped mode
        # if full_img.shape == (256,320): # full CRED 1 frame 
        #     sub_img = full_img[rc[0]:rc[1], rc[2]:rc[3]] # crop it to the beams local subframe (same reference frame as ref_x, ref_y pixels)

        # else: # CRED 1 must be in cropped mode. Crop mode only crops y (as of 13/9/25)
        #     y_offset = int( 256 - full_img.shape[0] )
        #     sub_img = full_img[rc[0]-y_offset : rc[1]-y_offset, rc[2] : rc[3]]
        # # could detect and interpolate bad pixels - only if this becomes a problem 
    
        mea_x, mea_y, a, b, theta, pupil_mask = self.detect_pupil( sub_img ,
                                                                  sigma=2, 
                                                                  threshold=0.5, 
                                                                  plot=False, 
                                                                  savepath=self.savepath)

        return( mea_x, mea_y, a, b, theta, pupil_mask )


    def calculate_pixel_offsets( self ):
        ref_x, ref_y, _ = self.extract_pupil_references(  ) # rc is the subframe pixels [r1,r2,c1,c2] in global frame coord

        mea_x, mea_y, a_semiMaj, b_semiMin, theta, pupil_mask = self.measure_pupil_center( ) 

        if abs( abs(a_semiMaj) - abs(b_semiMin) ) > 5: # semi major axis is 5 pixels bigger than semi minor
            x_pixel_offsets = 0
            y_pixel_offsets = 0
            print( f"WARNING - highly elliptical. Semi major axis = {np.round(a,2)}pixels which > 5 pixels of semi minor axis = {np.round(b,2)}pixels. Not safe to apply offset to DL VCM. Try manually adjusting")
        else:
            # calculate pixel offsets
            x_pixel_offsets = ref_x - mea_x
            y_pixel_offsets = ref_y - mea_y


        return x_pixel_offsets, y_pixel_offsets
    



    def _send_offsets_to_mcs(self, x_offset, y_offset, mode='bright'):

        """
        convert pixel offsets to millimeters 
        note matrix may need additional calibration onskjy

        Parameters
        ----------
        x_offset, : float
        y_offset : float
        mode : string ('bright'|'faint')

        """


        #x_off, y_off = pixel_offsets
        if 'bright' in mode.lower():
            # note pix_to_mm could be different for each beam. For baldr the VCMs lie in same plane so should be fine as I
            pix_to_mm = np.array([[1,0],[0,1]]) * 18 / 16 # 18mm beam over 16 pixel (upgraded baldr, old baldr was 12) diameter 
        elif 'faint' in mode.lower():
            pix_to_mm = np.array([[1,0],[0,1]]) * 18 / 6 # 18mm beam over 6 pixel diameter 
        else:
            raise UserWarning( f"mode must be bright or faint. We failed on this case. mode = {mode.lower()} " )
        millimeter_offsets = pix_to_mm @ [x_offset, y_offset]
        
        # wag expects to receive a vector of values [beam1,beam2,beam3,beam4]
        # if we only want to update a single parameter we can give a range (0,0) for beam 1, (1,1) beam 2 ect upto (3,3) for beam 4
        # HERE WE ONLY EVER UPDATE ONE BEAM AT A TIME!   

        # mcs deals with this now 
        #beam_range = f"({int(self.beam)-1}:{int(self.beam)-1})" # we specify for only one beam !

        x_offset = millimeter_offsets[0] #offset are flipped! x on the detector is y on sky
        y_offset = millimeter_offsets[1] 

        delta_pixel_lim = 10 # limit of pixels to move in 1 iteration (baldr frame)
        x_offset = np.round( np.clip(x_offset, -18 / 16 * delta_pixel_lim, 18 / 16 * delta_pixel_lim), 3)
        y_offset = np.round( np.clip(y_offset, -18 / 16 * delta_pixel_lim, 18 / 16 * delta_pixel_lim), 3)
        
        msg = {
            "origin": "s_bld_pup_autoalign_sky",
            "beam":int(self.beam)-1,
            "data": [
                {"bld_x_pup_offset": -x_offset}, # delta_x = ref-pos, if delta_x is neg we want to move to the left when looking at camera frame, which is +ve in VCM land 
                {"bld_y_pup_offset": -y_offset}, # delta_y = ref-pos, if delta_y is neg we want to move to the left when looking at camera frame, which is +ve in VCM land 
                {"bld_complete": 1},
            ],
        }
        #print(msg) ### Debug test here 
        self.send_and_recv_ack(msg)

        ########## OLD mcs format direct wag format  
        # ## Send x offset
        # msg = {
        #     "cmd": "s_bld_pup_autoalign_sky",
        #     "data": [
        #         {"name": "bld_x_pup_offset"},
        #         {"range": beam_range},
        #         {"value": [x_offset]},
        #     ]
        # }
        
        # self.send_and_recv_ack(msg)


        # #time.sleep(0.05)
        
        # ## Send y offset
        # msg = {
        #     "cmd": "s_bld_pup_autoalign_sky",
        #     "data": [
        #         {"name": "bld_y_pup_offset"},
        #         {"range": beam_range},
        #         {"value": [y_offset]},
        #     ]
        # }

        # self.send_and_recv_ack(msg)
        ############################## end old rts 

    def send_and_recv_ack(self, msg):
        # recieve ack
        print(f"sending {msg}")
        resp = self.mcs_client.send_payload(msg, decode_ascii=False)
        if resp is None or resp.get("ok") == False:
            print(resp)
            print("Failed to send offsets to MCS. Is the MCS running?")
        else:
            print("msg acked")


    def test_mcs(self):
        
        msg = {
            "origin": "s_bld_pup_autoalign_sky",
            "beam":int(self.beam)-1, 
            "data": [
                {"bld_x_pup_offset": x_offset},
                {"bld_y_pup_offset": y_offset},
                {"bld_complete": 1},
            ],
        }

        self.send_and_recv_ack(msg)
    
        # beam_range = f"({int(self.beam)-1}:{int(self.beam)-1})" # we specify for only one beam ! 
        # ## Send x offset
        # msg = {
        #     "cmd": "s_bld_pup_autoalign_sky",
        #     "data": [
        #         {"name": "bld_x_pup_offset"},
        #         {"range": beam_range},
        #         {"value": [np.random.uniform()]},
        #     ]
        # }
        
        #self.send_and_recv_ack(msg)

        # ## Send y offset
        # msg = {
        #     "cmd": "s_bld_pup_autoalign_sky",
        #     "data": [
        #         {"name": "bld_y_pup_offset"},
        #         {"range": beam_range},
        #         {"value": [np.random.uniform()]},
        #     ]
        # }


        # self.send_and_recv_ack(msg)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Autoalign Baldr beams.")
    parser.add_argument(
        "--beam",
        required=True,
        type=str,
        choices=['1','2','3','4'],
        help="which beam to align",
    )
    

    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=['bright','faint'],
        help="are we in Baldrs bright (12 pixel diam per pupil) or faint (6 pixel diam per pupil)?",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="path/to/baldr/rtc/config/toml/file (default: None, in which case we find it the active one)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default='mcs', # dont do internal here , this is done else where to deal with phasemasks too
        help="what is the output of this script? default is just to send offset info to mcs",
    )

    parser.add_argument(
        "-s",
        "--savepath",
        type=str,
        default="None",
        help="Path to save the alignment results (default: current directory)",
    )

    parser.add_argument(
        "--test_mcs",
        action="store_true",
        default=False,
        help="Do we test the mcs (don't calculate on image stream just send the correct dict format). Default False",
    )


    args = parser.parse_args()

    baldr_aa = BaldrAA( beam = args.beam, 
                       mode = args.mode,
                       config_filepath=args.config_file, 
                       output=args.output, 
                       savepath = args.savepath,
                       )

    #x_offset, y_offset = baldr_aa.calculate_pixel_offsets( )

    if args.test_mcs : 
        baldr_aa.test_mcs()

    else:
        if baldr_aa.output == "mcs":
            x_offset, y_offset = baldr_aa.calculate_pixel_offsets()
            #print(x_offset, y_offset)
            baldr_aa._send_offsets_to_mcs(x_offset, y_offset, mode = baldr_aa.mode)

        elif baldr_aa.output == 'internal':
            raise NotImplementedError("Internal alignment is done with another script in asgard alignment")
            # could do baldr_aa.move_pupil_with_phasemasks( ) <- something like this is already implemented in 
            # eng gui and asgard-alginment/common/m_bld_align.py script
            # risk is if peiple apply this in the wrong circumstance phaseamsk could
            

        else:
            raise UserWarning(f"output method: {baldr_aa.output} does not exist. Try output = 'mcs'.")

