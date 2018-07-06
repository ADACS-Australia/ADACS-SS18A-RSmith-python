"""Code for generating log messages

Developed with the following library versions:

    python:        3.5.2
"""

# For legacy-Python compatability
from __future__ import print_function

import sys
import time

class log_stream(object):
    """This class  manages the formatting of log output."""

    def __init__(self):
        self.t_start     = time.time()
        self.t_last      = self.t_start
        self.indent_size = 3
        self.n_indent    = 0
        self.hanging     = False
        self.set_fp()

    def set_fp(self,fp_out=None):
        if(fp_out==None):
            self.fp = sys.stderr
        else:
            self.fp = fp_out

    def unhang(self):
        """If the log did not end previously with a carriage return, add one."""
        if(self.hanging):
            print ('',file=self.fp)
            return True
        else:
            return False

    def indent(self):
        """Compute the next indent."""
        print (self.indent_size*self.n_indent*' ',end='', file=self.fp)
        self.fp.flush()
        
    def open(self,msg):
        """Open a new indent bracket for the log."""
        self.unhang()
        self.indent()
        print(msg,end='', file=self.fp)
        self.fp.flush()
        self.hanging=True
        self.n_indent+=1
        self.t_last=time.time()
    
    def append(self,msg):
        """Add to the end of the current line in the log."""
        print (msg,end='', file=self.fp)
        self.fp.flush()
        self.hanging=True

    def comment(self,msg):
        """Add a one-line comment to the log."""
        self.unhang()
        self.indent()
        print (msg,end='', file=self.fp)
        self.fp.flush()
        self.hanging=True

    def raw(self,msg):
        """Print raw, unformatted text to the log."""
        self.unhang()
        print (msg, file=self.fp)
        self.fp.flush()
        self.hanging=False
    
    def close(self,msg=None,time_elapsed=False):
        """Close a new indent bracket for the log.  Add an elapsed time since
        the last open to the end if time_elapsed=True"""
        self.n_indent-=1
        if(msg!=None):
            if(not self.hanging):
                self.indent()
            if(time_elapsed):
                dt         =time.time()-self.t_last
                msg_time   =" (%d seconds)"%(dt)
            else:
                msg_time=''
            print (msg+msg_time, end='\n', file=self.fp)
            self.fp.flush()
            self.hanging=False
    
    def error(self,err_msg,code=None):
        """Emit an error message and exit."""
        self.unhang()
        if(code):
            message = err_msg+" [code="+code+"]"
        else:
            message = err_msg
        raise Exception(message)

# Initialize the log stream
log=log_stream()
