# spawns a process using Popen that runs t_signit_process.py
# then sends a SIGINT to it after a few seconds

import os
import signal
import subprocess
import time
import sys


def main():
    print("Starting signit manager...")
    # Start the dummy signit process
    process = subprocess.Popen(
        [sys.executable, "tests/t_signit_process.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Let it run for a few seconds
        time.sleep(3)
        #print("Sending SIGINT to signit process...")
        #os.killpg(os.getpgid(process.pid), signal.SIGINT)

        # Wait a moment to see if the process has been terminated
        #time.sleep(1)

        # Check the return code to confirm termination
        #if process.poll() is not None:
        #    print(f"Process has been terminated with return code: {process.poll()}")
        #else:
        #    print("Process is still running, SIGINT may have been ignored.")

        # # Send SIGINT to the process
        process.send_signal(signal.SIGINT)

        # Wait for the process to complete and get output
        stdout, stderr = process.communicate()
        print("Signit process output:")
        print(stdout)
        if stderr:
            print("Signit process errors:")
            print(stderr)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if process.poll() is None:
            print("Terminating signit process...")
            process.terminate()
            process.wait()
        print("Signit manager finished.")

if __name__ == "__main__":
    main()
