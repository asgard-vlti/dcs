#!/usr/bin/env python3

""" ====================================================================
At least a temporary solution to avoid data headers passed by WAG being
overwritten over the course of multiple nights.

The intent:

This watchdog monitors for the content of /data/headers/
only acts when new *.iss or *.biniss are written there
and dispatches them to the appropriate (date) directories

This was turned into a service that is defined in:
/etc/systemd/system/wag_dispatch_header.service

-Frantz.
==================================================================== """

import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler as fseHandler
from datetime import datetime

def dispatch_header(fpath):
    # destination directory
    now = datetime.utcnow()
    ddir = f"/data/{now.year}{now.month:02d}{now.day:02d}/headers/"
    if not os.path.exists(ddir):
        os.makedirs(ddir)
        print(f"Destination directory {ddir} was created.")

    os.rename(fpath, ddir+fpath.split('/')[-1])
    print(f"Header {fpath} dispatched.")

class ChangeHandler(fseHandler):
    def on_created(self, event):
        if event.is_directory:
            return # ignore directory events

        _, fext = os.path.splitext(event.src_path)
        if fext in ['.iss', '.biniss']:
            dispatch_header(event.src_path)

        else:
            print(f"Not doing anything with {event.src_path}")

def monitor_directory(path):
    hdlr = ChangeHandler()
    observer = Observer()
    observer.schedule(hdlr, path, recursive=False)
    observer.start()
    print(f"Monitoring directory: {path}")

    while True:
        time.sleep(1)
    ### try:
    ###     while True:
    ###         time.sleep(1)
    ### except KeyboardInterrupt:
    ###     observer.stop()
    observer.stop()
    observer.join()

if __name__ == "__main__":
    dir_path = "/data/headers/"
    monitor_directory(dir_path)
