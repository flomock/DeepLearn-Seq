import numpy as np


def flatten(l):
    '''
    Given a list, possibly nested to any level, return it flattened.
    '''
    result = []
    for item in l:
        if isinstance(item, list):
            # http://bit.ly/2biB44i
            # http://bit.ly/2b3Nwa4
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def ohe(str):
    '''
    ...
    '''
    if str == 'A':
        return [1, 0, 0, 0]
    if str == 'C':
        return [0, 1, 0, 0]
    if str in ('T', 'U'):  # DNA, RNA
        return [0, 0, 1, 0]
    if str == 'G':
        return [0, 0, 0, 1]
    else:
        return [0, 0, 0, 0]


def ohe_matrix(M):
    '''
    # np.nan
    X = np.array([
        ['A', 'C', 'R'],
        ['A', 'Y', 'W']
        ])
    # we want [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    '''
    # arr = []
    # for i in range(M.shape[0]):  # loop over columns
    #     l = []
    #     for letter in M[i, :]:
    #         l.append(ohe(letter))
    #     arr.append(flatten(l))
    #     if i==500:
    #         print('hello')
    #     if i==2449:
    #         print('hello')
    # return np.array(arr)

    # arr = np.array(M,dtype='|S10')
    '''better version
    arr = np.empty([M.shape[0],M.shape[1],4], dtype=int)
    arr[M == 'A'] = [1,0,0,0]
    arr[M == 'C'] = [0,1,0,0]
    arr[M == 'T'] = [0,0,1,0]
    arr[M == 'U'] = [0,0,1,0]
    arr[M == 'G'] = [0,0,0,1]
    arr[M == '-'] = [0,0,0,0]
    newarr = arr.reshape(M.shape[0],M.shape[1]*4)
    # print(arr)
    return newarr
    '''
    # new version
    arr = np.zeros([M.shape[0], M.shape[1]], dtype=int)
    # arr[:,:] = 0
    arr[M == 'A'] = 1 #[1, 0, 0, 0]
    arr[M == 'C'] = 2 #[0, 1, 0, 0]
    arr[M == 'T'] = 3 #[0, 0, 1, 0]
    arr[M == 'U'] = 3 #[0, 0, 1, 0]
    arr[M == 'G'] = 4 #[0, 0, 0, 1]
    arr[M == '-'] = 0 #[0, 0, 0, 0]
    # newarr = arr.reshape(M.shape[0], M.shape[1] * 4)
    # print(arr)
    return arr