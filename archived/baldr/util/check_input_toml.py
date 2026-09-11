#!/usr/bin/env python3
import re
import sys
import os
import toml

def determine_dimensions(obj):
    """
    For a list, return 1 if flat (1D) or 2 if it is a list of lists (2D).
    """
    if not isinstance(obj, list):
        return None
    if len(obj) == 0:
        return 1
    if isinstance(obj[0], list):
        return 2
    return 1

def get_structure_signature(obj):
    """
    Recursively compute a structure signature for the given object.
    
    - If the object is a dict, return a dict mapping keys to signatures.
    - If it is a list, return a tuple ("list", dimension, signature of first element or None).
    - For numbers, treat int and float as the same ("number").
    - Otherwise, return the type name.
    """
    if isinstance(obj, dict):
        return {k: get_structure_signature(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        dim = determine_dimensions(obj)
        if len(obj) == 0:
            inner = None
        elif dim == 1:
            inner = get_structure_signature(obj[0])
        elif dim == 2:
            inner = get_structure_signature(obj[0][0]) if obj[0] else None
        return ("list", dim, inner)
    else:
        if isinstance(obj, (int, float)):
            return "number"
        return type(obj).__name__

def compare_signatures(sig1, sig2, path=""):
    """
    Recursively compare two structure signatures.
    
    Returns a list of error messages; an empty list means the signatures match.
    """
    errors = []
    if isinstance(sig1, dict) and isinstance(sig2, dict):
        for k in sig1:
            new_path = f"{path}.{k}" if path else k
            if k not in sig2:
                errors.append(f"Missing key '{new_path}' in candidate configuration")
            else:
                errors.extend(compare_signatures(sig1[k], sig2[k], new_path))
        for k in sig2:
            if k not in sig1:
                new_path = f"{path}.{k}" if path else k
                errors.append(f"Extra key '{new_path}' found in candidate configuration")
    else:
        if sig1 != sig2:
            errors.append(f"Type/dimension mismatch at '{path}': expected {sig1}, got {sig2}")
    return errors

def get_first_beam_config(config):
    """
    Look for the first key in the config that matches 'beam\d+'.
    Return the nested configuration under that key.
    """
    for key in config:
        if re.match(r'^beam\d+', key):
            return config[key]
    return None

def validate_two_files(ref_file, candidate_file):
    """
    Load two TOML files, extract their beam configuration (ignoring the beam number)
    and compare their structure.
    Returns a list of errors (empty if compatible).
    """
    if not os.path.exists(ref_file):
        return [f"Reference file not found: {ref_file}"]
    if not os.path.exists(candidate_file):
        return [f"Candidate file not found: {candidate_file}"]
    
    ref_config = toml.load(ref_file)
    cand_config = toml.load(candidate_file)
    
    ref_beam = get_first_beam_config(ref_config)
    if ref_beam is None:
        return [f"No beam configuration (e.g., 'beam1') found in reference file."]
    
    cand_beam = get_first_beam_config(cand_config)
    if cand_beam is None:
        return [f"No beam configuration (e.g., 'beam1') found in candidate file."]
    
    ref_sig = get_structure_signature(ref_beam)
    cand_sig = get_structure_signature(cand_beam)
    
    errors = compare_signatures(ref_sig, cand_sig, "beam")
    return errors



#------------------------------------------------------------------
def get_shape(obj):
    """
    If obj is a list of lists, return (rows, cols).
    If obj is a flat list, return (len(obj), 1).
    Otherwise, return None.
    """
    if not isinstance(obj, list):
        return None
    if len(obj) == 0:
        return (0, 0)
    if isinstance(obj[0], list):
        rows = len(obj)
        cols = len(obj[0])
        return (rows, cols)
    else:
        return (len(obj), 1)

def check_matrix_dimensions(cfg):
    """
    Given a beam configuration loaded from the TOML file, extract the key matrices
    and check their dimensions for compatibility with the RTC multiplication:
    
    We require:
    
    (a) I2A: shape (m, n)  — multiplying by a flattened image of length n produces an m-vector.
    (b) I0_dm and norm_pupil_dm: shape (m, 1).
    (c) I2M_LO and I2M_HO: shape (p, m) such that (I2M_* * sig) is a p-vector.
    (d) M2C_LO and M2C_HO: shape (q, p) so that their products yield DM command vectors of size (q, 1), and these must be compatible.
    
    Returns a list of error messages (empty list if no issues found).
    """
    errors = []
    
    try:
        ctrl_model = cfg["H3"]["ctrl_model"]
    except KeyError:
        errors.append("Missing key: H3.ctrl_model")
        return errors
    
    # try:
    #     ref_pupils = cfg["H3"]["reference_pupils"]
    # except KeyError:
    #     errors.append("Missing key: H3.reference_pupils")
    #     return errors

    # Get matrices (they should be represented as lists of lists if matrices; or lists if vectors)
    I2A = ctrl_model.get("I2A")
    I2M_LO = ctrl_model.get("I2M_LO")
    I2M_HO = ctrl_model.get("I2M_HO")
    M2C_LO = ctrl_model.get("M2C_LO")
    M2C_HO = ctrl_model.get("M2C_HO")
    I0 = ctrl_model.get("I0")
    norm_pupil = ctrl_model.get("norm_pupil")
    # (If the key for the normalized pupil on DM is named differently, adjust accordingly)
    
    # Check existence:
    if I2A is None:
        errors.append("Missing I2A matrix in ctrl_model")
    if I0 is None:
        errors.append("Missing I0 vector in reference_pupils")
    if norm_pupil is None:
        errors.append("Missing norm_pupil vector in reference_pupils")
    if I2M_LO is None:
        errors.append("Missing I2M_LO matrix in ctrl_model")
    if I2M_HO is None:
        errors.append("Missing I2M_HO matrix in ctrl_model")
    if M2C_LO is None:
        errors.append("Missing M2C_LO matrix in ctrl_model")
    if M2C_HO is None:
        errors.append("Missing M2C_HO matrix in ctrl_model")
    


    # If any are missing, return errors.
    if errors:
        return errors
    
    # Get shapes.
    shape_I2A = get_shape(I2A)
    shape_I2M_LO = get_shape(I2M_LO)
    shape_I2M_HO = get_shape(I2M_HO)
    shape_M2C_LO = get_shape(M2C_LO)
    shape_M2C_HO = get_shape(M2C_HO)
    
    # Check I2A dimensions.
    if shape_I2A is None:
        errors.append("I2A is not in valid list format.")
    else:
        m, n = shape_I2A

        try: 
            I0_dm = I2A @ I0
            norm_pupil_dm = I2A @ norm_pupil
            shape_I0_dm = get_shape(I0_dm)
            shape_norm = get_shape(norm_pupil_dm)

            if shape_I0_dm != (m, 1):
                errors.append(f"I0_dm shape mismatch: expected ({m},1), got {shape_I0_dm}.")
            if shape_norm != (m, 1):
                errors.append(f"norm_pupil_dm shape mismatch: expected ({m},1), got {shape_norm}.")
                    
        except:
            print( "issues with I2M @ I0 or I2M @ norm_pupil")
            errors.append("issues with I2M @ I0 or I2M @ norm_pupil")
        # For image multiplication, the image (flattened) should have size n,
        # and I0_dm and norm_pupil_dm should be (m, 1).

    # Check I2M_LO and I2M_HO: they must have shape (p, m) for some p.
    if shape_I2M_LO is None:
        errors.append("I2M_LO is not in valid list format.")
    else:
        p1, m1 = shape_I2M_LO
        if m1 != shape_I2A[0]:
            errors.append(f"I2M_LO inner dimension mismatch: expected {shape_I2A[0]}, got {m1}.")
    
    if shape_I2M_HO is None:
        errors.append("I2M_HO is not in valid list format.")
    else:
        p2, m2 = shape_I2M_HO
        if m2 != shape_I2A[0]:
            errors.append(f"I2M_HO inner dimension mismatch: expected {shape_I2A[0]}, got {m2}.")
    
    # Check M2C_LO and M2C_HO: they multiply a p-vector (from I2M_*) to produce DM commands.
    if shape_M2C_LO is None:
        errors.append("M2C_LO is not in valid list format.")
    else:
        q1, r1 = shape_M2C_LO
        if r1 != shape_I2M_LO[0]:
            errors.append(f"M2C_LO inner dimension mismatch: expected {shape_I2M_LO[0]}, got {r1}.")
    
    if shape_M2C_HO is None:
        errors.append("M2C_HO is not in valid list format.")
    else:
        q2, r2 = shape_M2C_HO
        if r2 != shape_I2M_HO[0]:
            errors.append(f"M2C_HO inner dimension mismatch: expected {shape_I2M_HO[0]}, got {r2}.")
    
    # Optionally: Check that DM command components (c_LO and c_HO computed in RTC)
    # would have the same shape, i.e. q1 and q2 should match.
    if shape_M2C_LO is not None and shape_M2C_HO is not None:
        if shape_M2C_LO[0] != shape_M2C_HO[0]:
            errors.append(f"DM command dim ensions mismatch: M2C_LO produces {shape_M2C_LO[0]} rows but M2C_HO produces {shape_M2C_HO[0]} rows.")

    return errors


def main():
    if len(sys.argv) != 3:
        print("Usage: check_two_beam_toml.py <reference_file.toml> <candidate_file.toml>")
        sys.exit(1)
    
    ref_file = sys.argv[1]
    candidate_file = sys.argv[2]
    
    errors = validate_two_files(ref_file, candidate_file)
    if errors:
        print("Structural incompatibilities found:")
        for err in errors:
            print(" -", err)
        #sys.exit(1)
    else:
        print("The candidate configuration is structurally compatible with the reference.")



    config = toml.load(candidate_file)
    
    beam_conf = get_first_beam_config(config)
    if beam_conf is None:
        print("No beam configuration (e.g., 'beam1') found in file.")
        sys.exit(1)
    
    # Run the matrix dimension checks on the beam configuration.
    errors = check_matrix_dimensions(beam_conf)
    if errors:
        print("Matrix dimension errors found:")
        for err in errors:
            print(" -", err)
        sys.exit(1)
    else:
        print("Matrix dimensions are compatible for RTC multiplications.")

    print("here")
    

if __name__ == "__main__":
    main()


#  python check_input_toml.py /home/asg/Progs/repos/dcs/baldr/baldr_config_1.toml /home/asg/Progs/repos/dcs/baldr/baldr_config_2.toml