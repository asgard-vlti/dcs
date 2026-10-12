"""Consistent UTC timestamps for DCS service logs."""

import logging
import time


class UTCFormatter(logging.Formatter):
    """Render record times in UTC with milliseconds and an explicit Z suffix."""

    converter = time.gmtime
    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03dZ"
