#!/usr/bin/env python3
import sys
import random
import toml

def random_vector(length):
    """Generate a flattened 1D list of random floats."""
    return [random.uniform(0, 1) for _ in range(length)]


def random_matrix(rows, cols):
    """Generate a 2D list (matrix) with random floats."""
    return [[random.uniform(0, 1) for _ in range(cols)] for _ in range(rows)]


def generate_config(beam_id, phase_key, P, M):
    config = {}
    
    # Use beam-specific key.
    beam_key_str = f"beam{beam_id}"
    config[beam_key_str] = {}
    
    config[beam_key_str][phase_key] = {}
    config[beam_key_str][phase_key]["ctrl_model"] = {
        # State fields:
        "DM_flat": "baldr",
        "signal_space": "pixel",
        "LO": 2,
        "controller_type": "PID",
        "inverse_method_LO": "pinv",
        "inverse_method_HO": "zonal",
        "auto_close": 0,
        "auto_open": 1,
        "auto_tune": 0,
        "bias": random_matrix(1, P)[0],  # Flattened 1D array of size P.
        "dark": random_matrix(1, P)[0],  # Flattened 1D array of size P.
        "close_on_strehl_limit": 0.5,
        "open_on_strehl_limit": 0.1,
        "open_on_flux_limit": 0.2,
        "open_on_dm_limit": 0.2,
        "LO_offload_limit": 0.2,
        # Pixel indices (for bdr_pixels)
        "crop_pixels": [1,2,3,4],
        "pupil_pixels": list(range(12,20)),
        "bad_pixels": list(range(22,30)),
        "interior_pixels": list(range(30,40)),
        "secondary_pixels": list(range(40,50)),
        "exterior_pixels": list(range(50,60)),
        # filters
        "bad_pixel_mask": [1 for __ in range(P)],    
        "pupil": [1 for __ in range(P)],   
        "secondary": [1 for __ in range(P)],   
        "exterior": [1 for __ in range(P)],   
        "inner_pupil_filt": [1 for __ in range(P)],   
        # Matrices
        "szm": M,
        "sza": 144,
        "szp": P,
        "I2A": random_matrix(144, P),
        "I2M": random_matrix(M, 144),
        "I2M_LO": random_matrix(M, 144),
        "I2M_HO": random_matrix(M, 144),
        "M2C": random_matrix(144, M),
        "M2C_LO": random_matrix(144, M),
        "M2C_HO": random_matrix(144, M),
        "I2rms_sec": random_matrix(2, 2),
        "I2rms_ext": random_matrix(2, 2),
        # Reference intensities:
        "I0": random_matrix(1, P)[0],
        "N0": random_matrix(1, P)[0],
        "norm_pupil": random_matrix(1, P)[0],
        # Camera config:
        "camera_config": {
            "fps": "30.0",
            "gain": "1.5",
            "testpattern": "none",
            "bias": "none",
            "flat": "none",
            "imagetags": "none",
            "led": "none",
            "events": "none",
            "extsynchro": "none",
            "rawimages": "none",
            "cooling": "none",
            "mode": "auto",
            "resetwidth": "none",
            "nbreadworeset": "none",
            "cropping": "none",
            "cropping_columns": "none",
            "cropping_rows": "none",
            "aduoffset": "none"
        },
    }
    
    return config

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python generate_fake_config.py <beam_id> <phase_key> <P> <M>")
        sys.exit(1)
    
    beam_id = sys.argv[1]
    phase_key = sys.argv[2]
    P = int(sys.argv[3])
    M = int(sys.argv[4])
    
    config = generate_config(beam_id, phase_key, P, M)
    filename = f"baldr_config_{beam_id}.toml"
    with open(filename, "w") as f:
        toml.dump(config, f)
    
    print(f"Fake config file generated: {filename}")