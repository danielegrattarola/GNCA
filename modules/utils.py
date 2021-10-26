def is_iterable(x):
    try:
        _ = iter(x)
        if isinstance(x, dict):
            return False
        return True
    except TypeError:
        return False
