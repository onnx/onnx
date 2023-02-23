import unittest
from functools import wraps


def has_onnxruntime():
    try:
        import onnxruntime  # pylint: disable=W0611

        del onnxruntime

        return True
    except ImportError:
        return False


def skip_if_no_onnxruntime(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not has_onnxruntime():
            raise unittest.SkipTest("onnxruntime not installed")
        fn(*args, **kwargs)

    return wrapper
