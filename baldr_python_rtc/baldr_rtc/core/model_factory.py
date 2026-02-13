

from baldr_python_rtc.baldr_rtc.core.state import RTCModel
from baldr_python_rtc.baldr_rtc.rtc.controllers import build_controller
from baldr_python_rtc.baldr_rtc.core.state import BDRConfig
import numpy as np

def build_rtc_model(cfg:BDRConfig) -> RTCModel:
    st = cfg.state
    space = (st.signal_space or "pix").strip().lower()

    I2A = np.asarray(cfg.matrices.I2A, dtype=float) # if space == "dm" else None

    I2M_LO = np.asarray(cfg.matrices.I2M_LO, dtype=float)
    I2M_HO = np.asarray(cfg.matrices.I2M_HO, dtype=float)
    M2C_LO = np.asarray(cfg.matrices.M2C_LO, dtype=float)
    M2C_HO = np.asarray(cfg.matrices.M2C_HO, dtype=float)

    # filters 
    inner_pupil_filt = np.asarray( cfg.filters.inner_pupil_filt, dtype=bool).reshape(-1) 
    
    # NEW (NOT LEGACY)
    strehl_filt =  np.asarray( cfg.filters.strehl_filt, dtype=bool).reshape(-1) 

    interc = float( cfg.filters.opd_m_interc)
    slope_1 = float( cfg.filters.opd_m_slope_1)
    slope_2 = float( cfg.filters.opd_m_slope_2)
    x_knee =  float(  cfg.filters.opd_m_x_knee )


    # reduction 
    #dark = np.asarray(cfg.matrices.M2C_HO, dtype=float)
    # references from legacy toml
    I0 = np.asarray(cfg.reference_pupils.I0, dtype=float).reshape(-1)
    N0 = np.asarray(cfg.reference_pupils.N0, dtype=float).reshape(-1)
    dark = np.asarray(cfg.reference_pupils.dark, dtype=float).reshape(-1)
    if space == "dm":
        # NOTE: assumes I0/N0 are already in the SAME reduced pixel vector space as I2A expects
        I0 = I2A @ I0
        N0 = I2A @ N0
        inner_pupil_filt = (I2A @ inner_pupil_filt).astype(bool)
        strehl_filt = (I2A @ strehl_filt).astype(bool)
    i_setpoint_runtime = I0 / np.mean( N0[inner_pupil_filt]  ) 
    N0_runtime = np.mean( N0[inner_pupil_filt]  ) #N0
    



    # controllers
    n_LO = int(I2M_LO.shape[0])
    n_HO = int(I2M_HO.shape[0])

    #Try speak to camera to get fps
    try:
        import zmq
        cam_server = f"tcp://192.168.100.2:6667"
        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.REQ)
        # timeouts so we don't hang forever
        s.setsockopt(zmq.RCVTIMEO, 5000)   # 5s receive timeout
        s.setsockopt(zmq.SNDTIMEO, 5000)   # 5s send timeout
        s.connect(cam_server)

        s.send_string("get_fps")
        fps = float( s.recv().decode("ascii") )
        dt = 1/fps

    except:
        dt = 1/1000 
        print("Could not connect to camera server - stting default dt=1kHz")    

    # The control loop.py currently does not use the controller class
    # everything is hard coded in the hot loop. The only thing the controller is used for is storing and resetting 
    # gains - so as long as the gain interface is consistent this is fine. Later we should port to use for the 
    #controller abstract classes and also remove dt dependance.
    ct = "leaky"  #cfg.state.controller_type.strip().lower()
    if ct == "pid":
        ctrl_LO = build_controller("pid", n_LO, dt=dt, kp=0, ki=0, kd=0, u_min=None, u_max=None)
        ctrl_HO = build_controller("pid", n_HO, dt=dt, kp=0, ki=0, kd=0, u_min=None, u_max=None)
    elif ct == "leaky":
        ctrl_LO = build_controller("leaky", n_LO, dt=dt, rho=0.98, ki=0.05, kp=0, u_min=None, u_max=None)
        ctrl_HO = build_controller("leaky", n_HO, dt=dt, rho=0.97, ki=0.0, kp=0, u_min=None, u_max=None)
    else:
        print(f"controller_type {ct} is not implemented\n!!!!!!!!!! JUT CONTINUE WITH Leaky , FIX THIS LATER")
        ctrl_LO = build_controller("leaky", n_LO, dt=dt,rho=0.98, ki=0.05, kp=0, u_min=None, u_max=None)
        ctrl_HO = build_controller("leaky", n_HO, dt=dt,rho=0.97, ki=0.0, kp=0, u_min=None, u_max=None)
    
    return RTCModel(
        signal_space=space,
        I2A=I2A,
        I2M_LO=I2M_LO,
        I2M_HO=I2M_HO,
        M2C_LO=M2C_LO,
        M2C_HO=M2C_HO,
        N0_runtime=N0_runtime,
        dark=dark.reshape(32,32),
        inner_pupil_filt=inner_pupil_filt,
        strehl_filt = strehl_filt,
        i_setpoint_runtime=i_setpoint_runtime,
        ctrl_LO=ctrl_LO,   # or g.ctrl_LO if you store controllers in globals
        ctrl_HO=ctrl_HO,
        process_frame=None,
        perf_model=None,
        perf_param=(interc, slope_1, slope_2, x_knee),
    )

