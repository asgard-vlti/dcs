from mcs_client import ZmqReq

z = ZmqReq("tcp://192.168.100.2:6667")

reply = z.send_payload("status", is_str=True, decode_ascii=False, jsonloads=False)
# z.s.send_string("status")
# print("sent")
# reply = z.s.recv_string()
print(reply)
print("done")