#!/bin/bash

# downloads the log files from wag on mimir

rsync asg@wag:/vltdata/tmp/wag.*.log wag_logs/
rsync asg@wag:/vltdata/tmp/bob.wv*.log /data/wag_logs/
echo "Log files from WAG copied over to /data/wag_logs/"
