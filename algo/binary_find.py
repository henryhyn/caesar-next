"""
循环递增序列查找某个数
"""


def binary_find(arr, val):
    start = 0
    end = len(arr) - 1
    while start <= end:
        mid = start + (end - start) // 2
        if val == arr[mid]:
            return mid

        if arr[start] < arr[mid]:
            if arr[start] <= val < arr[mid]:
                end = mid - 1
            else:
                start = mid + 1
        else:
            if arr[mid] < val <= arr[end]:
                start = mid + 1
            else:
                end = mid - 1
    return -1


if __name__ == '__main__':
    for i in range(11):
        print(i, binary_find([6, 8, 9, 2, 4, 5], i))
