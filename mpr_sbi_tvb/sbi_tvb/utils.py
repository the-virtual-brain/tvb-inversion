import functools


def custom_setattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(custom_getattr(obj, pre) if pre else obj, post, val)


def custom_getattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
