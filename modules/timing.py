# libs
from time import time 
from tabulate import tabulate
import pandas as pd
import re

# classes
class Timing:
    def __init__(self):
        self.t = {}
        self.r = {}
        self.start_time = time()

    def start(self, s):
        if self.existing(s):
            assert(self.running(s) == False)
            self.t[s] -= time()
        else:
            self.t[s] = -time()
        self.r[s] = True

    def stop(self, s):
        assert(self.existing(s))
        self.t[s] += time()
        self.r[s] = False

    def existing(self, s):
        return(s in self.r.keys())

    def running(self, s):
        assert(s in self.r.keys())
        return(self.r[s])

    def __str__(self):
        total_time = time() - self.start_time
        tags = []
        times_sec = []
        times_per = []
        for k in self.t.keys():
            if not self.running(k):
                tags += [k]
                times_sec += [self.t[k]]
                times_per += [100 * self.t[k] / total_time]
        df = pd.DataFrame({'tag' : tags, 'time[s]': times_sec, 'time[%]': times_per})
        s = tabulate(df, headers = 'keys', tablefmt = 'psql', showindex=False)
        s1 = s.splitlines()[0]
        s2 = ''.join([' ']*(len(s1)-9)) + '|\n'
        s = s1 + '\n| timing' + s2 + s

        return(s)

# global variables
t = Timing()


