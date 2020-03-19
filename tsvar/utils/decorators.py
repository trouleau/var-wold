from functools import wraps


def enforce_observed(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if self._observed:
            return fn(self, *args, **kwargs)
        else:
            raise Exception(('Model is not properly set. '
                             '`set_data` must be called first.'))
    return wrapped
