from __future__ import annotations

import queue
import threading
import time
import numpy as np

from baldr_python_rtc.baldr_rtc.core.state import MainState, RuntimeGlobals, ServoState
from baldr_python_rtc.baldr_rtc.telemetry.ring import TelemetryRingBuffer




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
                i_raw = np.zeros((1, 1), dtype=np.float32)
            else:
                fr = self.g.camera_io.get_frame( ) #reform=True)
                i_raw = fr.data - self.g.model.dark
                #print(i_raw)
            
        
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
            #opd_est = self.g.model.perf_model( i_norm , self.g.model.perf_param)
             
            s = i_norm  - self.g.model.i_setpoint_runtime 

            # project intensity signal to error in modal space 
            e_LO = self.g.model.I2M_LO @ s 
            e_HO = self.g.model.I2M_HO @ s

            # control signals
            u_LO = self.g.model.ctrl_LO.process( e_LO )
            u_HO = self.g.model.ctrl_HO.process( e_HO )
            
            # Project mode to DM commands 
            c_LO = self.g.model.M2C_LO @ u_LO 
            c_HO = self.g.model.M2C_HO @ u_HO

            dcmd = c_LO + c_HO #+ c_LO_inj + c_HO_inj

            # TODO: replace metrics with real computation
            i_in_pup_tmp = np.mean( self.g.telem_ring.i_space[:,self.g.model.inner_pupil_filt] , axis=1) # used to calculate SNR
            snr_metric = np.mean( i_in_pup_tmp  ) / np.std( i_in_pup_tmp ) if np.std( i_in_pup_tmp ) !=0 else 0
            opd_metric = self.g.model.perf_model(i_norm, self.g.model.perf_param) if self.g.model.perf_model is not None else 0.0 # g.perf_funct( )

            # --- IO: write DM command (placeholder) ---
            if self.g.dm_io is not None:
                #cmd = np.zeros(140)
                #cmd[65] = 0.2
                # replace with real computed command vector
                self.g.dm_io.write(dcmd)

            # self.telem_ring.push(
            #     frame_id=self._frame_id,
            #     t_s=t_now,
            #     lo_state=int(self.g.servo_mode_LO),
            #     ho_state=int(self.g.servo_mode_HO),
            #     paused=bool(self.g.pause_rtc),
            #     metric_flux=metric_flux,
            #     metric_strehl=metric_strehl,
            # )


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

            next_t += dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

