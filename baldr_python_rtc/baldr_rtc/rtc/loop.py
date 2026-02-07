from __future__ import annotations

import queue
import threading
import time
import numpy as np

from baldr_python_rtc.baldr_rtc.core.state import MainState, RuntimeGlobals, ServoState
from baldr_python_rtc.baldr_rtc.telemetry.ring import TelemetryRingBuffer


# opd model
def piecewise_continuous(x, interc, slope_1, slope_2, x_knee):
    # piecewise linear (hinge) model 
    return interc + slope_1 * x + slope_2 * np.maximum(0.0, x - x_knee)


interc = 9368.549647307767
slope_1 = -5882.950106515396
slope_2 = 4678.104756734429
x_knee = 1.5324802815558276

# update N0_runtime onsky (not N0)
# get clear pupil avg

# update N0 in g
# also g.N0_runtime 
# make sure consistent with baldr_python_rtc/baldr_rtc/server.build_rtc_model



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
            print( 'here to' )
            self.update_N0_runtime()


        elif t == "SET_LO_GAIN":
            lo_gain = cmd.get("gain")
            self.g.rand = lo_gain[0]
            print( cmd, lo_gain[0] )


        # elif t == "CLOSE_LO":
        #     close_LO = 1
        # elif t == "OPEN_LO":
        #     close_LO = 0


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
        #close_LO = 0 
        
        # UPDATE N0
        # print('update N0_runtime before beginning. Put on clear pupil')
        # input('press enter when ready')
        # self.update_N0_runtime()

        lo_gain = 0.2 # default start 
        # start
        while not self.stop_event.is_set():
            if self.g.servo_mode == MainState.SERVO_STOP:
                self.stop_event.set()
                break

            ## Uncommet and debug this!!
            self._drain_commands()

            if self.g.pause_rtc:
                time.sleep(0.01)
                continue

            self._frame_id += 1
            t_now = time.time()

            # --- IO: read camera frame ---
            if self.g.camera_io is None:
                # fallback: dummy frame (keeps thread alive)
                i_raw0 = np.zeros((1, 1), dtype=np.float32)
            else:
                fr = self.g.camera_io.get_frame( ) #reform=True)
                i_raw0 = fr.data - self.g.model.dark
                #print(i_raw)
            
            ## Trying slow 
            no_2_avg = 100
            avg_cnt = 0
            run_iteration = 0
            img_list = []
            while avg_cnt < no_2_avg :
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
                
                # use self.g.model.process_frame to take moving average or something 
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
                try:
                    #if close_LO :
                    if self.g.servo_mode_LO:
                        u_LO = 0.95 * u_LO - lo_gain * e_LO #self.g.model.ctrl_LO.process( e_LO )
                        
                    else:
                        u_LO = 0.0 * u_LO - 0.0*e_LO

                    if self.g.servo_mode_HO:
                        # implement close_HO later 
                        u_HO = 0.90 * u_HO - 0.05*e_HO #self.g.model.ctrl_HO.process( e_HO )
                    else:
                        u_HO = 0.0 * u_HO - 0.0*e_HO
                except:
                    print('here')
                    u_LO = np.zeros_like(e_LO)
                    u_HO = np.zeros_like(e_HO)
                # Project mode to DM commands 
                c_LO = self.g.model.M2C_LO @ u_LO 
                c_HO = self.g.model.M2C_HO @ u_HO

                dcmd = c_LO + c_HO #+ c_LO_inj + c_HO_inj

                # TODO: replace metrics with real computation
                i_in_pup_tmp = np.mean( self.g.telem_ring.i_space[:,self.g.model.inner_pupil_filt] , axis=1) # used to calculate SNR
                snr_metric = np.mean( i_in_pup_tmp  ) / np.std( i_in_pup_tmp ) if np.std( i_in_pup_tmp ) !=0 else 0

                # if working (params are hard coding)
                opd_sig_tmp  = np.mean( i[ self.g.model.strehl_filt] ) / self.g.model.N0_runtime 
                opd_metric = 0.03 * piecewise_continuous( opd_sig_tmp , interc, slope_1, slope_2, x_knee) 
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


                if self.telem_ring is not None and self.g.model is not None:
                    self.telem_ring.push(
                        frame_id=self._frame_id,
                        t_s=t_now,
                        lo_state=int(self.g.servo_mode_LO.value),
                        ho_state=int(self.g.servo_mode_HO.value),
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

