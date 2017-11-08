from itertools import zip_longest


def chunk(n, iterable, fillvalue=None):
    '''
    Generate sequences of `chunk_size` elements from `iterable`.

    source: stackoverflow, 8991506

    Usage:
    grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    '''

    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)
