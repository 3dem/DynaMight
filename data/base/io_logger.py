#!/usr/bin/env python

"""
Simple I/O pipe logging to file
Can replace stdout to log all output
"""

import sys

class IOLogger(object):
    def __init__(self, filename="Default.log", quiet=False):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.quiet = quiet

    def write(self, message):
        if not self.quiet:
            self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        if not self.quiet:
            self.terminal.flush()
        self.log.flush()