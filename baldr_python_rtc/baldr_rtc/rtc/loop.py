from __future__ import annotations

import queue
import threading
import time
import numpy as np

from astropy.io import fits
from pathlib import Path
import datetime 
from xaosim.shmlib import shm
from baldr_python_rtc.baldr_rtc.core.state import MainState, RuntimeGlobals, ServoState
from baldr_python_rtc.baldr_rtc.telemetry.ring import TelemetryRingBuffer

# global number of frames to average before correction
#N0_2_AVG = 1

# opd model (used for performance monitoring and lucky imaging for updating ZWFS intensity setpoint onsky)
def piecewise_continuous(x, interc, slope_1, slope_2, x_knee):
    # piecewise linear (hinge) model 
    return interc + slope_1 * x + slope_2 * np.maximum(0.0, x - x_knee)

# #{'interc': 629.1738798737395, 'slope_1': -2032.353630969202, 'slope_2': 1208.6141949891808, 'x_knee': 0.13625865519335179, 'cost': 1} 
# interc = 629.173#9368.549647307767
# slope_1 = -2032.35 #-5882.950106515396
# slope_2 = 1208.614 #4678.104756734429
# x_knee = 0.1362 #1.5324802815558276


    

# help to apply "gain" editing commands 
def _apply_gain(ctrl, param: str, idx, value: float):
    if not hasattr(ctrl, param):
        print(f"controller has no '{param}'")
        return
    
    arr = getattr(ctrl, param)
    if arr is None:
        print(f"controller '{param}' is None")
        return

    if idx == "all":
        arr[:] = value
    else:
        # bounds check
        idx = list(idx)
        n = int(arr.size)
        if any((i < 0 or i >= n) for i in idx):
            print(f"index out of range for '{param}': n={n}, idx={idx[-1]}")
            return
        arr[idx] = value



class RTCThread(threading.Thread):
    def __init__(
        self,
        globals_: RuntimeGlobals,
        command_queue: "queue.Queue[dict]",
        telem_ring: TelemetryRingBuffer,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.g = globals_
        self.command_queue = command_queue
        self.telem_ring = telem_ring
        self.stop_event = stop_event
        self._frame_id = 0


    def update_N0_runtime(self):
        self._apply_command({'type':"PAUSE"})
        img_list = []
        for _ in range(100) :
            fr = self.g.camera_io.get_frame( ) 
            i_raw0 = fr.data - self.g.model.dark
            img_list.append( i_raw0 )
            time.sleep(0.01)
        
        N0_new = np.mean( img_list, axis = 0).reshape(-1)
        if self.g.model.signal_space == "dm":
            # we project onto DM actuators (dm space)
            N0_new = self.g.model.I2A @ N0_new
 
        N0_runtime = np.mean( N0_new[self.g.model.inner_pupil_filt]  ) 

        print('previous N0_runtime = ', self.g.model.N0_runtime)
        print('candidate N0_runtime = ', N0_runtime)

        usr_input = input('type 1 to update N0_runtime? ')
        if usr_input == '1':
            self.g.model.N0_runtime = N0_runtime
        
        usr_input = input('before resuming put mask back in (later this will be automatic..).\npress enter when ready to start loop')
        self._apply_command({'type':"RESUME"})


    def update_I0_runtime(self):
        # from telemetry buffer store N samples (while loop is runnin) 
        # of performance metric (opd_metric) dark subtracted image
        # after filled N samples , quantile images based on performance metric
        # (with opd metric we want lower quantile => higher strehl)
        # then i_setpoint_new = <I0>/N0_runtime 
        # take Bayesian weights of new image  


        def _lucky_img(I0_meas,
                       image_processing_fn , 
                       performance_model ,
                       model_param,
                       quantile_threshold=0.03, 
                       keep="<threshold"):
            # I0_meas is list of 2D images, we process them with image_processing_fn (user specifiied function)
            # the input that processed signal to performance_model (another user specified function, with model_param input)
            # we then do a percentile cut at quantile_threshold and keep images based on "keep" input (i.e. images with performance metric less than threshold)

            img_signal = np.array( [image_processing_fn(ii) for ii in I0_meas])

            perf_est = performance_model(img_signal, **model_param)

            perf_threshold = np.quantile( perf_est , quantile_threshold)

            if keep == "<threshold":
                I0_lucky = np.array( I0_meas )[ perf_est < perf_threshold ]
            elif keep == ">threshold":
                I0_lucky = np.array( I0_meas )[ perf_est > perf_threshold ]
            else:
                raise UserWarning("_lucky_img has invalid 'keep' entry. Must be keep = '<threshold'|'>threshold'")    
            return I0_lucky 
        
        
        def _image_processing_fn(i, filt=self.g.model.strehl_filt):
            return( np.mean( i[filt] ))
        
        ######### START 
        I_prior = self.g.model.i_setpoint_runtime

        perf_quant_threshold = 0.03 # quantil cut for performance metric (OPD here)
        # need to find a way to do this without disturbing the loop - because we want to do this ideally in closed loop
        N_dumps = 5
        sleep_between_dumps = 1.0
        samp = 0 
        i_norm_samples = []
        perf_metric_samples = []
        print(f"dumping telem ring every {sleep_between_dumps} second(s)\nto update ZWFS intensity setpoint\n-----------")
        while samp < N_dumps:
            print(f"...telem ring dump {samp}/{N_dumps}")
            # could we just dump a few ring buffers (need to be synchronized the dumping)
            # normalize so in the sasme normalized space as self.g.model.i_setpoint_runtime 
            i_norm = self.telem_ring.i_space / self.g.model.N0_runtime
            i_norm_samples.append( i_norm )

            # perf_metric_samples.append( self.telem_ring.opd_metric )  
            time.sleep( sleep_between_dumps )
            samp += 1

        # flatten 
        i_norm_samples = np.array([item[0] for item in i_norm_samples]) 
        #perf_metric_samples = np.array([item[0] for item in perf_metric_samples])

        # this is hard coded for now so fragile!
        model_param_tmp = {
            "interc":self.g.model.perf_param[0],
            "slope_1":self.g.model.perf_param[1],
            "slope_2":self.g.model.perf_param[2],
            "x_knee":self.g.model.perf_param[3],
        }
        
        # list of our lucky zwfs images 
        lucky_imgs = _lucky_img( I0_meas = i_norm,
                       image_processing_fn = _image_processing_fn, 
                       performance_model = piecewise_continuous ,
                       model_param = model_param_tmp, 
                       quantile_threshold=perf_quant_threshold , 
                       keep="<threshold")
        
        # pixelwise avg lucky imgs 
        I_meas = np.mean(lucky_imgs, axis=0 )

        # pixelwise std err 
        sigma_meas = np.std( lucky_imgs ,axis = 0) #/ np.sqrt( len(lucky_imgs) )

        alpha = 1 # # 0.001 strong prior, 0.01 weak prior
        sigma_prior = alpha * np.mean( I_prior ) #np.quantile( I_prior ,0.95) # we could make this more advance considering pixelwise prior weighting  

        # bayesian weigths (gaussian noise)
        w_meas = 1/sigma_meas**2
        w_prior = 1/sigma_prior**2

        # Bayesian pixelwise update 
        I_post = (w_meas * I_meas + w_prior * I_prior) / (w_meas + w_prior)

        # update 
        self.g.model.i_setpoint_runtime = I_post

        # save for sanity checking 
        path_tmp = self._write_I0_update_fits(I_prior, I_post)

        print( f"completed update of i_setpoint_runtime.\nsaved fits with before & after i_setpoint_runtime for comparison here:\n{path_tmp}\n")


    # thinking more this should turn into a more general write fits for comparison file
    def _write_I0_update_fits(self, prior_I0, post_I0, outdir=None):
        """
        Save prior/post i_setpoint_runtime as a FITS with 2 image HDUs.
        prior_I0, post_I0: array-like (1D or 2D). Stored as float32.
        """
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Choose output directory
        if outdir is None:
            # Prefer your existing telemetry/log directory if you have it
            tstamp_rough = datetime.datetime.now().strftime("%Y-%m-%d")
            outdir = f"/home/asg/ben_feb2026_bld_telem/{tstamp_rough}/beam{self.g.beam}/"#getattr(self.g, "telem_dir", None) or getattr(self.g, "log_dir", None) or "/tmp"
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        beam_id = self.g.beam #getattr(self, "beam_id", getattr(self.g, "beam_id", None))
        phasemask = self.g.phasemask
        # Convert to arrays (preserve 2D if already 2D; otherwise keep as 1D)
        prior = np.asarray(prior_I0, dtype=np.float32)
        post  = np.asarray(post_I0,  dtype=np.float32)

        # File name 
        fname = outdir / f"I0_runtime_update_beam{beam_id}__mask-{phasemask}_{ts.replace(':','-')}.fits"

        # ---- primary header metadata ----
        hdr = fits.Header()
        hdr["DATE-OBS"] = (ts, "UTC timestamp of update")
        if beam_id is not None:
            hdr["BEAM"] = (int(beam_id), "Baldr beam id")
        hdr["SRC"] = ("update_I0_runtime", "Source routine")

        # Optional camera/system settings (only if available)
        # (Safely fill if your object has these fields)
        #fps  = getattr(self.g, "fps", None) or getattr(getattr(self.g, "model", None), "fps", None)
        #gain = getattr(self.g, "gain", None) or getattr(getattr(self.g, "model", None), "gain", None)

        #if fps is not None:  hdr["FPS"]  = (float(fps),  "Camera FPS")
        #if gain is not None: hdr["GAIN"] = (float(gain), "Camera gain")

        hdr["NAXPRIOR"] = (int(prior.ndim), "Dims of PRIOR array")
        hdr["NAXPOST"]  = (int(post.ndim),  "Dims of POST array")
        hdr["NPRIOR"]   = (int(prior.size), "Number of elements in PRIOR")
        hdr["NPOST"]    = (int(post.size),  "Number of elements in POST")

        phdu = fits.PrimaryHDU(header=hdr)
        hdu_prior = fits.ImageHDU(data=prior, name="PRIOR_I0")
        hdu_post  = fits.ImageHDU(data=post,  name="POST_I0")

        fits.HDUList([phdu, hdu_prior, hdu_post]).writeto(fname, overwrite=True)
        return str(fname)


    def update_KL(self, N_dumps=5, k_use=40, savefits=True):


        # Build a KL/PCA rotation in the *current HO modal subspace* using telemetry in signal space (s).
        # User is responsible for putting the system in a sensible state before calling this.
        #
        # We use: e_HO[t] = I2M_HO @ s[t]
        # Then PCA on e_HO telemetry gives an orthonormal rotation V such that:
        #   m_KL = V^T m_old
        # and we update matrices consistently:
        #   I2M_HO <- V^T @ I2M_HO
        #   M2C_HO <- M2C_HO @ V
        #
        # Optional command fields:
        #   n_samples : number of most recent samples to use (default: all available in ring)
        #   k_modes   : number of KL modes to keep/rotate (default: current HO mode count)


        sleep_between_dumps = 1.0
        samp = 0 
        signal_samples = []
        perf_metric_samples = []
        print(f"dumping telem ring every {sleep_between_dumps} second(s)\nto update ZWFS intensity setpoint\n-----------")
        while samp < N_dumps:
            print(f"...telem ring dump {samp}/{N_dumps}")
            # could we just dump a few ring buffers (need to be synchronized the dumping)
            # normalize so in the sasme normalized space as self.g.model.i_setpoint_runtime 

            signal_samples.append( np.array(self.telem_ring.s,copy=True) )
            # so we can filter 
            perf_metric_samples.append( np.array(self.telem_ring.opd_metric,copy=True) )  
            time.sleep( sleep_between_dumps )
            samp += 1

        # filter for the best quality signals (to avoid regimes where sensor is non-linear)
        signal_samples_flat = np.array([item[0] for item in signal_samples]) 
        perf_metric_samples_flat = np.array([item[0] for item in perf_metric_samples])

        perf_filt = perf_metric_samples_flat < np.quantile( perf_metric_samples_flat , 0.05) 

        S = signal_samples_flat[ perf_filt ]

        I2M_old = np.asarray(self.g.model.I2M_HO)
        M2C_old = np.asarray(self.g.model.M2C_HO)

        k_total = int(I2M_old.shape[0])

        k_use = min(k_use, k_total) # clamp 

        # --- build HO modal telemetry from signal telemetry ---
        # e_HO[t] = I2M_old @ s[t]  =>  E = S @ I2M_old.T
        E = S @ I2M_old.T  # (T, k_total)

        # --- demean + covariance over chosen subset ---
        E = E[:, :k_use]
        E = E - np.mean(E, axis=0, keepdims=True)
        C = (E.T @ E) / max(1, (E.shape[0] - 1))  # (k_use, k_use)

        # --- eigen-decompose covariance (symmetric) ---
        evals, evecs = np.linalg.eigh(C)          # ascending
        order = np.argsort(evals)[::-1]           # descending
        evals = evals[order]
        V = evecs[:, order]                       # (k_use, k_use), columns are eigenvectors

        # --- update matrices (explicit) ---
        I2M_new = I2M_old.copy()
        M2C_new = M2C_old.copy()

        # rotate the first k_use modes; leave the rest unchanged
        I2M_new[:k_use, :] = V.T @ I2M_old[:k_use, :]
        M2C_new[:, :k_use] = M2C_old[:, :k_use] @ V

        self.g.model.I2M_HO = I2M_new
        self.g.model.M2C_HO = M2C_new

        print(f"BUILD_KL: updated HO basis using {E.shape[0]} samples, k_use={k_use}/{k_total}")
        print(f"BUILD_KL: top eigenvalues (variance) = {evals[:min(8, len(evals))]}")


        tstamp_rough = datetime.datetime.now().strftime("%Y-%m-%d")
        outdir = f"/home/asg/ben_feb2026_bld_telem/{tstamp_rough}/beam{self.g.beam}/"#getattr(self.g, "telem_dir", None) or getattr(self.g, "log_dir", None) or "/tmp"
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        beam_id = self.g.beam #getattr(self, "beam_id", getattr(self.g, "beam_id", None))
        phasemask = self.g.phasemask

        ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # File name 
        fname = outdir / f"I0_runtime_update_beam{beam_id}__mask-{phasemask}_BUILD_KL_{ts.replace(':','-')}.fits"

        # ---- primary header metadata ----
        hdr = fits.Header()
        hdr["DATE-OBS"] = (ts, "UTC timestamp of update")
        if beam_id is not None:
            hdr["BEAM"] = (int(beam_id), "Baldr beam id")
        hdr["SRC"] = ("update_KL", "Source routine")

        # Optional camera/system settings (only if available)
        # (Safely fill if your object has these fields)
        #fps  = getattr(self.g, "fps", None) or getattr(getattr(self.g, "model", None), "fps", None)
        #gain = getattr(self.g, "gain", None) or getattr(getattr(self.g, "model", None), "gain", None)

        #if fps is not None:  hdr["FPS"]  = (float(fps),  "Camera FPS")
        #if gain is not None: hdr["GAIN"] = (float(gain), "Camera gain")

        # hdr["NAXPRIOR"] = (int(prior.ndim), "Dims of PRIOR array")
        # hdr["NAXPOST"]  = (int(post.ndim),  "Dims of POST array")
        # hdr["NPRIOR"]   = (int(prior.size), "Number of elements in PRIOR")
        # hdr["NPOST"]    = (int(post.size),  "Number of elements in POST")

        phdu = fits.PrimaryHDU(header=hdr)

        things_2_write = {
            "S":S,
            "C":C,
            "V":V,
            "M2C_old":M2C_old,
            "M2C_new":M2C_new,
            "I2M_old":I2M_old,
            "I2M_new":I2M_new,
        }

        fits_list = [fits.PrimaryHDU(header=hdr)]
        for name,thing in things_2_write.items():
            fits_list.append( fits.ImageHDU(data=thing, name=f"{name}") )


        fits.HDUList(fits_list).writeto(fname, overwrite=True)
        return str(fname)




    def _apply_command(self, cmd: dict) -> None:
        t = cmd.get("type", "")
        if t == "PAUSE":
            self.g.pause_rtc = True
        elif t == "RESUME":
            self.g.pause_rtc = False
        elif t == "STOP":
            self.g.servo_mode = MainState.SERVO_STOP
        elif t == "SET_LO":
            self.g.servo_mode_LO = ServoState(int(cmd["value"]))
        elif t == "SET_HO":
            self.g.servo_mode_HO = ServoState(int(cmd["value"]))
        elif t == "SET_LOHO":
            self.g.servo_mode_LO = ServoState(int(cmd["lo"]))
            self.g.servo_mode_HO = ServoState(int(cmd["ho"]))
        elif t == "SET_TELEM":
            self.g.rtc_config.state.take_telemetry = bool(cmd.get("enabled", False))
        elif t == "LOAD_CONFIG":
            new_cfg = cmd.get("rtc_config")
            if new_cfg is not None:
                self.g.rtc_config = new_cfg
                self.g.active_config_filename = cmd.get("path", self.g.active_config_filename)
        
        elif t == "UPDATE_N0_RUNTIME":
            #print( 'here to' )
            self.update_N0_runtime()


        elif t == "UPDATE_I0_RUNTIME":
            #print('here I0')
            self.update_I0_runtime()

        elif t == "UPDATE_RECON_LO":
            print( 'to do here' )
            # move pahsemask out (usr responsibility)
            # get <N0> in signal space (use self.telem_ring.i_space), 
            # normalize <N0> to mask in [0,1], 
            # g.model.I2M_LO = (g.model.I2M_LO.T * <N0>).T            

            I2M_LO_before = self.g.model.I2M_LO
            avg_i_space = np.mean(self.telem_ring.i_space, axis = 0)
            # normalize mask [0-1]
            avg_i_space_norm = (avg_i_space-np.max( avg_i_space )) / (np.max(avg_i_space)-np.min(avg_i_space))
            self.g.model.I2M_LO = (I2M_LO_before * avg_i_space_norm)


            # save for sanity checking 
            path_tmp = self._write_I0_update_fits(I2M_LO_before, self.g.model.I2M_LO)

            print( f"completed update of I2M_LO.\nsaved fits with before & after self.g.model.I2M_LO for comparison here:\n{path_tmp}\n")


        elif t == "UPDATE_RECON_HO":
            print('to do here')
            # move pahsemask out (usr responsibility)
            # get <N0> in signal space (use self.telem_ring.i_space), 
            # normalize <N0> to mask in [0,1], 
            # g.model.I2M_HO = (g.model.I2M_HO.T * <N0>).T  

            I2M_HO_before = self.g.model.I2M_HO
            avg_i_space = np.mean(self.telem_ring.i_space, axis = 0)
            # normalize mask [0-1]
            avg_i_space_norm = (avg_i_space-np.max( avg_i_space )) / (np.max(avg_i_space)-np.min(avg_i_space))
            self.g.model.I2M_HO = (I2M_HO_before * avg_i_space_norm)

            # save for sanity checking 
            path_tmp = self._write_I0_update_fits(I2M_HO_before, self.g.model.I2M_HO)

            print( f"completed update of I2M_HO.\nsaved fits with before & after self.g.model.I2M_HO for comparison here:\n{path_tmp}\n")



        elif t == "FRAMES_2_AVG":
            #if cmd['value'] == ''
            #else try update 
            self.g.number_frames_2_avg = int(cmd['value'])

        elif t == "SET_LO_GAIN":
            _apply_gain(self.g.model.ctrl_LO, cmd["param"], cmd["idx"], float(cmd["value"]))

        elif t == "SET_HO_GAIN":
            _apply_gain(self.g.model.ctrl_HO, cmd["param"], cmd["idx"], float(cmd["value"]))

        elif t == "ZERO_GAINS":
            for ctrl in (self.g.model.ctrl_LO, self.g.model.ctrl_HO):
                for p in ("kp", "ki", "kd"):
                    if hasattr(ctrl, p):
                        arr = getattr(ctrl, p)
                        if arr is not None:
                            arr[:] = 0.0

        elif t == "WRITE_DM_FLAT":
            self.g.write_to_flat = True


        elif t== "UPDATE_RECON_KL":
            fname = self.update_KL( n_use = 1000,k_use=40 ,savefits = True)
            print(f"saved telemetry from UPDATE_RECON_KL here:\n{fname}")

    def _drain_commands(self) -> None:
        for _ in range(100):
            try:
                cmd = self.command_queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._apply_command(cmd)
            finally:
                self.command_queue.task_done()

    def run(self) -> None:

        # some quick 
        fps = float(self.g.rtc_config.fps) if self.g.rtc_config.fps > 0 else 1000.0
        dt = 1.0 / fps
        next_t = time.perf_counter()
        
        
        # opd model (from exterior pixels of ZWFS )
        interc, slope_1, slope_2, x_knee = self.g.model.perf_param
        # g.rtc_config.filters.opd_m_interc
        # g.rtc_config.filters.opd_m_slope_1
        # g.rtc_config.filters.opd_m_slope_2
        # g.rtc_config.filters.opd_m_x_knee

        # init controller feedback vectors 
        u_LO = np.zeros_like(self.g.model.ctrl_LO.ki)
        u_HO = np.zeros_like(self.g.model.ctrl_HO.ki)

        # UPDATE N0
        # print('update N0_runtime before beginning. Put on clear pupil')
        # input('press enter when ready')
        # self.update_N0_runtime()

        #lo_gain = 0.2 # default start 
        # start
        while not self.stop_event.is_set():
            if self.g.servo_mode == MainState.SERVO_STOP:
                self.stop_event.set()
                break

            #
            self._drain_commands()

            if self.g.pause_rtc:
                time.sleep(0.01)
                continue

            self._frame_id += 1
            t_now = time.time()

            TT0 = time.perf_counter()
            

            avg_cnt = 0
            run_iteration = 0
            img_list = []
            while avg_cnt < self.g.number_frames_2_avg :
                # --- IO: read camera frame ---
                if self.g.camera_io is None:
                    # fallback: dummy frame (keeps thread alive)
                    i_raw0 = np.zeros((1, 1), dtype=np.float32)
                else:
                    fr = self.g.camera_io.get_frame( ) #reform=True)
                    i_raw0 = fr.data - self.g.model.dark
                    #print(i_raw)
                    
                img_list.append( i_raw0 )
                avg_cnt += 1
                # manual timing
                next_t += dt
                sleep = next_t - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
            # set flags
            run_iteration = 1
            avg_cnt = 0

            i_raw = np.mean(img_list,axis=0)
            img_list = [] # restart img_list 
            if run_iteration:
                # do loop 

                # process (average or otherwise)
                # signal  (i_setpoint_runtime should always be in the right space, if change space (function) we must update it )
                if self.g.model.signal_space.lower().strip() == 'pix':
                    i_space = i_raw.reshape(-1)
                elif self.g.model.signal_space.lower().strip() == 'dm':
                    i_space = self.g.model.I2A @ i_raw.reshape(-1)
                else:
                    raise UserWarning("invalid signal_space. Must be 'pix' | 'dm'")
                
                # use self.g.model.process_frame to take advance proceesing of image
                i = i_space #self.g.model.process_frame( i_space , fn = None) # this could be a simple moving average . In my sim I think i did this in error space - but should it be here? LPF straight up 

                # normalized intensity 
                i_norm = i / self.g.model.N0_runtime 

                # opd estimate 
                #opd_est = np.mean( i[ self.g.model.inner_pupil_filt.astype(bool) ] ) / self.g.model.N0_runtime#self.g.model.perf_model( i_norm , self.g.model.perf_param)
                
                s = i_norm  - self.g.model.i_setpoint_runtime 

                # project intensity signal to error in modal space 
                e_LO = self.g.model.I2M_LO @ s 
                e_HO = self.g.model.I2M_HO @ s

                # control signals # get sign 
                #try:
                #if close_LO :
                if self.g.servo_mode_LO:
                    
                    #print('here')
                    #self.g.model.ctrl_LO.ki
                    u_LO = self.g.model.ctrl_LO.rho * u_LO - self.g.model.ctrl_LO.ki * e_LO #lo_gain * e_LO #self.g.model.ctrl_LO.process( e_LO )
                    
                else:
                    u_LO[:] = 0.0

                if self.g.servo_mode_HO:
                    # implement close_HO later 
                    #np.array( [0.05 if ii < 10 else 0 for ii in range(len(e_HO))] )
                    u_HO = self.g.model.ctrl_HO.rho * u_HO - self.g.model.ctrl_HO.ki*e_HO #self.g.model.ctrl_HO.process( e_HO )
                    #print( self.g.model.ctrl_LO.ki , self.g.model.ctrl_HO.rho)
                else:
                    u_HO[:] = 0.0
                # except:
                #     print('here')
                #     u_LO = np.zeros_like(e_LO)
                #     u_HO = np.zeros_like(e_HO)
                # Project mode to DM commands 
                c_LO = self.g.model.M2C_LO @ u_LO 
                c_HO = self.g.model.M2C_HO @ u_HO

                dcmd = c_LO + c_HO #+ c_LO_inj + c_HO_inj

                # TODO: replace metrics with real computation
                i_in_pup_tmp = np.mean( self.g.telem_ring.i_space[:,self.g.model.inner_pupil_filt] , axis=1) # used to calculate SNR
                snr_metric = np.mean( i_in_pup_tmp  ) / np.std( i_in_pup_tmp ) if np.std( i_in_pup_tmp ) !=0 else 0

                # if working (params are hard coding)
                opd_sig_tmp  = np.mean( i_norm[ self.g.model.strehl_filt] ) 
                opd_metric =  piecewise_continuous( opd_sig_tmp , interc, slope_1, slope_2, x_knee) 
                # otherwise 
                #opd_metric = np.mean( i[ self.g.model.inner_pupil_filt.astype(bool) ] ) / self.g.model.N0_runtime#self.g.model.perf_model( i_norm , self.g.model.perf_param)
                #self.g.model.perf_model(i_norm, self.g.model.perf_param) if self.g.model.perf_model is not None else 0.0 # g.perf_funct( )

                # --- IO: write DM command (placeholder) ---
                if self.g.dm_io is not None:
                    #cmd = np.zeros(140)
                    #cmd[65] = 0.2
                    # replace with real computed command vector
                    self.g.dm_io.write(dcmd)


                # set flag to not iterate (until we average the next frames)
                run_iteration = 0


                if self.g.write_to_flat:
                    
                    dm = shm( f'/dev/shm/dm{self.g.beam}disp00.im.shm' )
                    current_flat =  dm.get_data()

                    avg_dcmd = np.mean(self.telem_ring.cmd, axis = 0)
                    ## when we were doing it fast from intensity space 
                    dm.set_data(current_flat + avg_dcmd.reshape(12,12))

                    ## when we were doing it slow from intensity space 
                    #dm.set_data(current_flat + dcmd.reshape(12,12))

                    # need to reset controller state 
                    u_LO = np.zeros_like(u_LO)
                    u_HO = np.zeros_like(u_HO)
                    # open loops 
                    self.g.servo_mode_LO = 0
                    self.g.servo_mode_LO = 0
                    # reset flag 
                    self.g.write_to_flat = False
                    dm.close(erase_file=False)
                    print('updated DM {g.beam} flat, and reset baldr LO & HO feedback state')

                if self.telem_ring is not None and self.g.model is not None:
                    self.telem_ring.push(
                        frame_id=self._frame_id,
                        t_s=t_now,
                        lo_state=int(self.g.servo_mode_LO), #.value
                        ho_state=int(self.g.servo_mode_HO), #.value
                        paused=bool(self.g.pause_rtc),

                        snr_metric = snr_metric,
                        opd_metric = opd_metric, 

                        i_raw=i_raw.reshape(-1),
                        i_space=i_space.reshape(-1),
                        s=s.reshape(-1),

                        e_lo=e_LO,
                        e_ho=e_HO,
                        u_lo=u_LO,
                        u_ho=u_HO,
                        c_lo=c_LO,
                        c_ho=c_HO,
                        cmd=dcmd,

                        ctrl_state_lo=getattr(self.g.model.ctrl_LO, "state", None),
                        ctrl_state_ho=getattr(self.g.model.ctrl_HO, "state", None),
                    )
            #TT1 = time.perf_counter()
            #print(TT1-TT0)





            ## OLD SUDO CODE 

            # # process (average or otherwise)
            # # signal  (i_setpoint_runtime should always be in the right space, if change space (function) we must update it )
            # if self.g.model.signal_space.lower().strip() == 'pix':
            #     i_space = i_raw.reshape(-1)
            # elif self.g.model.signal_space.lower().strip() == 'dm':
            #     i_space = self.g.model.I2A @ i_raw.reshape(-1)
            # else:
            #     raise UserWarning("invalid signal_space. Must be 'pix' | 'dm'")
            
            # # use self.g.model.process_frame to take moving average or something 
            # i = i_space #self.g.model.process_frame( i_space , fn = None) # this could be a simple moving average . In my sim I think i did this in error space - but should it be here? LPF straight up 

            # # normalized intensity 
            # i_norm = i / self.g.model.N0_runtime 

            # # opd estimate 
            # #opd_est = np.mean( i[ self.g.model.inner_pupil_filt.astype(bool) ] ) / self.g.model.N0_runtime#self.g.model.perf_model( i_norm , self.g.model.perf_param)
             
            # s = i_norm  - self.g.model.i_setpoint_runtime 

            # # project intensity signal to error in modal space 
            # e_LO = self.g.model.I2M_LO @ s 
            # e_HO = self.g.model.I2M_HO @ s

            # # control signals
            # u_LO = self.g.model.ctrl_LO.process( e_LO )
            # u_HO = self.g.model.ctrl_HO.process( e_HO )
            
            # # Project mode to DM commands 
            # c_LO = self.g.model.M2C_LO @ u_LO 
            # c_HO = self.g.model.M2C_HO @ u_HO

            # dcmd = c_LO + c_HO #+ c_LO_inj + c_HO_inj

            # # TODO: replace metrics with real computation
            # i_in_pup_tmp = np.mean( self.g.telem_ring.i_space[:,self.g.model.inner_pupil_filt] , axis=1) # used to calculate SNR
            # snr_metric = np.mean( i_in_pup_tmp  ) / np.std( i_in_pup_tmp ) if np.std( i_in_pup_tmp ) !=0 else 0

            # # if working (params are hard coding)
            # opd_sig_tmp  = np.mean( i[ self.g.model.strehl_filt] ) / self.g.model.N0_runtime 
            # opd_metric = 0.03 * piecewise_continuous( opd_sig_tmp , interc, slope_1, slope_2, x_knee) 
            # # otherwise 
            # #opd_metric = np.mean( i[ self.g.model.inner_pupil_filt.astype(bool) ] ) / self.g.model.N0_runtime#self.g.model.perf_model( i_norm , self.g.model.perf_param)
            # #self.g.model.perf_model(i_norm, self.g.model.perf_param) if self.g.model.perf_model is not None else 0.0 # g.perf_funct( )

            # # --- IO: write DM command (placeholder) ---
            # if self.g.dm_io is not None:
            #     #cmd = np.zeros(140)
            #     #cmd[65] = 0.2
            #     # replace with real computed command vector
            #     self.g.dm_io.write(dcmd)




            # if self.telem_ring is not None and self.g.model is not None:
            #     self.telem_ring.push(
            #         frame_id=self._frame_id,
            #         t_s=t_now,
            #         lo_state=int(self.g.servo_mode_LO.value),
            #         ho_state=int(self.g.servo_mode_HO.value),
            #         paused=bool(self.g.pause_rtc),

            #         snr_metric = snr_metric,
            #         opd_metric = opd_metric, 

            #         i_raw=i_raw.reshape(-1),
            #         i_space=i_space.reshape(-1),
            #         s=s.reshape(-1),

            #         e_lo=e_LO,
            #         e_ho=e_HO,
            #         u_lo=u_LO,
            #         u_ho=u_HO,
            #         c_lo=c_LO,
            #         c_ho=c_HO,
            #         cmd=dcmd,

            #         ctrl_state_lo=getattr(self.g.model.ctrl_LO, "state", None),
            #         ctrl_state_ho=getattr(self.g.model.ctrl_HO, "state", None),
            #     )

            # next_t += dt
            # sleep = next_t - time.perf_counter()
            # if sleep > 0:
            #     time.sleep(sleep)

