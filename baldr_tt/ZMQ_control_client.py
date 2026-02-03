''' In ipython, import this then use the python command line. 

'''
import numpy as np
import matplotlib.pyplot as plt
import zmq
import json
import base64

server_port = 6660
context = zmq.Context()
socket = context.socket(zmq.REQ)
#socket.connect(f"tcp://192.168.100.2:{server_port}")
socket.connect(f"tcp://localhost:{server_port}")

print(f"ZMQ shell interface to talk to the heimdallr server on port {server_port}")

def send(cmd):
    ''' Send a command to the server and print the reply. '''
    socket.send_string(cmd)
    try:
        resp = json.loads(socket.recv().decode("ascii"))
    except:
        print("Error: no reply from server")
        return None
    return resp

def get_im(cmd):
    """ Get the power spectrum or other image data"""
    resp = send(cmd)

    if resp['type'] == 'complex':
        dtype = np.complex64
    elif resp['type'] == 'double':
        dtype = np.float64
    elif resp['type'] == 'float':
        dtype = np.float32
    elif resp['type'] == 'int32':
        dtype = np.int32
    elif resp['type'] == 'int64':
        dtype = np.int64
    elif resp['type'] == 'uint8':
        dtype = np.uint8
    elif resp['type'] == 'uint16':
        dtype = np.uint16
    elif resp['type'] == 'uint32':
        dtype = np.uint32
    elif resp['type'] == 'uint64':
        dtype = np.uint64
    elif resp['type'] == 'int8':
        dtype = np.int8
    elif resp['type'] == 'int16':
        dtype = np.int16
    elif resp['type'] == 'float16':
        dtype = np.float16
    else:
        raise ValueError(f"Unknown type {resp['dtype']}")
    dat = np.frombuffer(base64.b64decode(resp['message']), dtype=dtype)
    dat = dat.reshape((resp['szx'], resp['szy']))
    return dat

