from mcs_client import ZmqReq
import socket
import time

# DM
z = ZmqReq("tcp://192.168.100.2:6667")
reply = z.send_payload("status", is_str=True, decode_ascii=False)
print(reply)
print("done")


# wag wd


def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(("wag", port))
        print(result)
        return "open" if result == 0 else "closed"

print(check_port(7501))

# time.sleep(3)

z = ZmqReq("tcp://192.168.100.1:7051")
reply = z.send_payload({"BTT1": {"process":"open"}}, is_str=False, decode_ascii=False)
print(reply)
print("done")