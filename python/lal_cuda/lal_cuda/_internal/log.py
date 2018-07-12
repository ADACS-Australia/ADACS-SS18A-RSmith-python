# For legacy-Python compatability
from __future__ import print_function

import sys
import time
import types
import datetime


class log_stream(object):
    """This class  manages the formatting of log output."""

    def __init__(self):
        self.t_start = time.time()
        self.t_last = []
        self.indent_size = 3
        self.n_indent = 0
        self.hanging = False
        self.set_fp()

    def set_fp(self, fp_out=None):
        if(fp_out is None):
            self.fp = sys.stderr
        else:
            self.fp = fp_out

    def unhang(self):
        """If the log did not end previously with a carriage return, add
        one."""
        if(self.hanging):
            print ('', file=self.fp)
            self.hanging = False

    def indent(self, overwrite=False):
        """Compute the next indent."""
        if(overwrite):
            print ('\r', end='', file=self.fp)
        print (self.indent_size * self.n_indent * ' ', end='', file=self.fp)
        self.fp.flush()

    def open(self, msg):
        """Open a new indent bracket for the log."""
        self.unhang()
        self.indent()
        print(msg, end='', file=self.fp)
        self.fp.flush()
        self.hanging = True
        self.n_indent += 1
        self.t_last.append(time.time())

    def append(self, msg):
        """Add to the end of the current line in the log."""
        print (msg, end='', file=self.fp)
        self.fp.flush()
        self.hanging = True

    def comment(self, msg, overwrite=False, hang=False):
        """Add a one-line comment to the log."""
        if(not overwrite):
            self.unhang()
        self.indent(overwrite)
        if(hang):
            print (msg, end='', file=self.fp)
        else:
            print (msg, end='\n', file=self.fp)
        self.fp.flush()
        self.hanging = hang

    def raw(self, msg):
        """Print raw, unformatted text to the log."""
        self.unhang()
        print (msg, file=self.fp)
        self.fp.flush()
        self.hanging = False

    def close(self, msg=None, time_elapsed=False):
        """Close a new indent bracket for the log.

        Add an elapsed time since the last open to the end if
        time_elapsed=True
        """
        self.n_indent -= 1

        # Sanity checks
        if(self.n_indent < 0):
            self.error("Invalid log closure.  n_indent has dropped below zero.")
        if(len(self.t_last) == 0):
            self.error("Invalid log closure.  t_last entries have been exhausted.")

        # This must be called every time to keep number of entries correct
        dt = time.time() - self.t_last.pop()

        # Generate message
        if(msg is not None):
            if(time_elapsed):
                msg_time = " (%d seconds)" % (dt)
            else:
                msg_time = ''
            if(not self.hanging):
                self.indent()
            print (msg + msg_time, end='\n', file=self.fp)
            self.fp.flush()
            self.hanging = False

    def progress_bar(self, gen, count, *args, **kwargs):
        """Display a progress bar for a generator."""

        # Render counter
        width = 30
        msg_len_last = 0
        start_time = time.time()
        self.comment("[%s] Remaining:" % (' ' * width), hang=True)
        for iteration, result in enumerate(gen(*args, **kwargs)):
            fraction_complete = float(iteration + 1) / float(count)
            ticks = int(fraction_complete * float(width + 1))
            secs_elapsed = time.time() - start_time
            secs_estimate = int(secs_elapsed / fraction_complete)
            secs_remaining = secs_estimate - secs_elapsed
            if(secs_remaining > 0):
                msg = "[%s%s] Remaining: %s" % ('#' * ticks, ' ' * (width - ticks),
                                                str(datetime.timedelta(seconds=secs_remaining)).split('.')[0])
                msg_len = len(msg)

                # Make sure to blank-out any old underlying text
                if(msg_len < msg_len_last):
                    msg += ' ' * (msg_len_last - msg_len)
                msg_len_last = msg_len
                self.comment(msg, overwrite=True, hang=True)
        msg = "[%s%s] Time elapsed: %s" % ('#' * ticks, ' ' * (width - ticks),
                                           str(datetime.timedelta(seconds=secs_elapsed)).split('.')[0])
        msg_len = len(msg)
        if(msg_len < msg_len_last):
            msg += ' ' * (msg_len_last - msg_len)
        self.comment(msg, overwrite=True)

    def error(self, err_msg, code=None):
        """Emit an error message and exit."""
        self.unhang()
        if(code):
            message = err_msg + " [code=" + code + "]"
        else:
            message = err_msg
        raise Exception(message)
