def max_pool(arr, m):
    n = len(arr)
    max_val = max(arr[0:m])
    res = [max_val]
    for i in range(1, n - m + 1):
        out_val = arr[i - 1]
        in_val = arr[i + m - 1]
        if in_val >= max_val:
            max_val = in_val
        elif out_val == max_val:
            max_val = max(arr[i:i + m])
        res.append(max_val)
    return res


def max_pool_v2(arr, m):
    n = len(arr)
    queue = []
    res = []
    for i in range(n):
        while queue and arr[queue[-1]] < arr[i]:
            queue.pop()
        queue.append(i)
        if i - queue[0] >= m:
            queue.pop(0)
        if i >= m - 1:
            res.append(arr[queue[0]])
    return res


def max_pool2d(matrix, m):
    mat = [max_pool_v2(arr, m) for arr in matrix]
    mat = [max_pool_v2(arr, m) for arr in zip(*mat)]
    return [arr for arr in zip(*mat)]


if __name__ == '__main__':
    matrix = [[5, 3, 2, 1, 4, 5],
              [2, 3, 1, 4, 6, 7],
              [6, 2, 1, 4, 5, 6],
              [5, 6, 3, 2, 4, 1]]
    print(max_pool2d(matrix, 3))
