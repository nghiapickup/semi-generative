import traceback
import os


class BaseExceptionHandle(Exception):
    __doc__ = 'Handling Base exception' \
              'and print out traceback'

    def __init__(self):
        pass

    def recall_traceback(self, sys_exc_info):
        self.exc_type, self.exc_obj, self.exc_tb = sys_exc_info
        fname = os.path.split(self.exc_tb.tb_frame.f_code.co_filename)[1]
        traceback.print_stack()
        print('File "' + str(fname) + '", line ' + str(self.exc_tb.tb_lineno) + '", ' + str(self.exc_obj))

class DataSizeConstraint(BaseExceptionHandle):
    """This Error will be raised when Unlabeled data size was violated"""

    def __init__(self, message):
        super(DataSizeConstraint, self).__init__()


class DataTypeConstraint(BaseExceptionHandle):
    """This Error will be raised when input data type was violated"""

    def __init__(self, message):
        super(DataTypeConstraint, self).__init__()

class MismatchLengthComponentList(BaseExceptionHandle):
    """"This Error will be raised when length of component list is different with class number"""

    def __init__(self, message):
        super(MismatchLengthComponentList, self).__init__()


class ComponentCountIsList(BaseExceptionHandle):
    """"This Error will be raised when component_count is not a list"""

    def __init__(self, message):
        super(ComponentCountIsList, self).__init__()
