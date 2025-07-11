import ray

def sync_remote(fn):
    """
    Given a Ray remote function `fn`, return a callable that
    calls `fn.remote(*args, **kwargs)` and blocks on ray.get.
    """
    def wrapped(*args, **kwargs):
        return ray.get(fn.remote(*args, **kwargs))
    return wrapped