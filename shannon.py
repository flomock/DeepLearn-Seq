def shannon(counts, base=2):
    '''
    shamelessly stolen from:
    https://github.com/biocore/scikit-bio/blob/9dc60b4248912a4804c90d0132888d6979a62d51/skbio/diversity/alpha/_base.py#L833

    import numpy as np
    a = np.array([1,2,3,2])
    shannon(a)
    # 1.91

    Entropy can be normalized:
    http://math.stackexchange.com/questions/395121/how-entropy-scales-with-sample-size

    Why np.log(base)? because np has no built-in function to specify the base
    of the logarithm, so we use the "logarithm base change rule"
    (stackoverflow, 25169297).
    '''
    import numpy as np

    freqs = counts / counts.sum()
    nonzero_freqs = freqs[freqs.nonzero()]
    return -(nonzero_freqs * np.log(nonzero_freqs)).sum() / np.log(base)
