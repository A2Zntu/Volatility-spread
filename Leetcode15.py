# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 20:34:12 2018

@author: Evan
"""

def threeSum(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    if len(nums) < 3:
        return []
    nums.sort()
    res = set()
    for i, v in enumerate(nums[:-2]): #至少要3個數 所以第一個最多跑到倒數第三個

        if i >= 1 and v == nums[i-1]:
            continue
        d = {}
        for x in nums[i+1:]: #中間的樹
            if x not in d:
                d[-v-x] = 1
                print("v:%i"%v)
                print("x:%i"%x)
                print(d)
            else:
                res.add((v, -v-x, x))
    return list(map(list, res))

nums = [-1,0,1,2,-1,-4, 3]
nums.sort()
print(nums)
print(nums[:-2])
res = threeSum(nums)
print(res)