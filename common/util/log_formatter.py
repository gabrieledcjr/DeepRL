#!/usr/bin/env python3
import logging
import re

regex = re.compile(r"\x1b(\[.*?[@-~]|\].*?(\x07|\x1b\\))", re.UNICODE)
class LogFormatter(logging.Formatter):
    def format(self,record):
        msg = super(LogFormatter, self).format(record)
        return regex.sub("", msg)
