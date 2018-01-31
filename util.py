from __future__ import print_function, division

import time
import sys

class Logger(object):
    """Simple logging object"""
    def __init__(self, verbose=False):
        self.verbose = verbose

    def info(self, string):
        if self.verbose:
            print(string)
        sys.stdout.flush()


class Timer(object):
    # Simple timer
    def __init__(self, verbose=False):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def tick(self):
        elapsed = self.elapsed()
        self.restart()
        return elapsed


def exp_filename(opt, name):
    # Returns output file name based on experiment name
    return opt.output_dir + '/' + opt.exp_name + '_' + name

def exp_temp_filename(opt, name):
    # Returns temporary file name based on experiment name
    return opt.temp_dir + '/' + opt.exp_name + '_' + name


def loadtxt(filename):
    # Load text file into list of strings
    txt = []
    with open(filename, 'r') as f:
        for l in f:
            txt.append(l.strip())
    return txt

def savetxt(filename, txt):
    # Print list of strings to file
    with open(filename, 'w+') as f:
        for l in txt:
            print(l, file=f)
