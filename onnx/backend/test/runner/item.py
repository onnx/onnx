from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# A container that hosts the test function and the associated
# test item (ModelProto)


class TestItem(object):
    def __init__(self, func, proto):
        self.func = func
        self.proto = proto
