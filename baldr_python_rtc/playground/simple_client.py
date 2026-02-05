import zmq
import matplotlib.pyplot as plt 
import numpy as np

address = 'tcp://127.0.0.1:3001' # or whatever baldr rtc server is (its printed when the server is started)
context = zmq.Context.instance()
socket = context.socket(zmq.REQ)
socket.connect( address )

############################################################
#some examples
socket.send_string('status') # you can check on the server side if the command was recieved (it should print to screen)
response = socket.recv_string()
print(response)

socket.send_string('poll_telem {"n":2,"fields":["i_raw"]}') # get 2 (n) polls of i_raw telemetry field from baldr rtc server 
response = socket.recv_string()
print(response)
# to turn it back to a dict instead of string 
res_dict = eval(response.replace('true','True').replace('false','False'))
# lets look at it 
plt.figure()
plt.imshow( np.mean( res_dict['fields']['i_raw'], axis=0).reshape(32,32) )
plt.show()

# close a loop 
socket.send_string("close_baldr_LO")
response = socket.recv_string()
print(response)

# check the status has changed correctly 
socket.send_string('status') # you can check on the server side if the command was recieved (it should print to screen)
response = socket.recv_string()
print(response)
