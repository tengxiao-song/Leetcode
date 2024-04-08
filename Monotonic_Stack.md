# 单调栈

单调栈是一个栈，其中元素是递增或递减的。

什么时候用单调栈呢？

**通常是一维数组，要寻找任一个元素的右边或者左边第一个比自己大或者小的元素的位置，此时我们就要想到可以用单调栈了**。时间复杂度为O(n)。

# 739. 每日温度

[力扣题目链接](https://leetcode.cn/problems/daily-temperatures/)

请根据每日 气温 列表，重新生成一个列表。对应位置的输出为：要想观测到更高的气温，至少需要等待的天数。如果气温在这之后都不会升高，请在该位置用 0 来代替。

例如，给定一个列表 temperatures = [73, 74, 75, 71, 69, 72, 76, 73]，你的输出应该是 [1, 1, 4, 2, 1, 1, 0, 0]。

```py
class Solution(object):
    def dailyTemperatures(self, temperatures):
        stack = []
        res = [0] * len(temperatures)
        for i in range(len(temperatures)):
            if not stack or temperatures[i] < temperatures[stack[-1]]:
                stack.append(i)  #存放的是index
            else:
                while stack and temperatures[i] > temperatures[stack[-1]]:  #持续比较
                    index = stack.pop()
                    res[index] = i - index
                stack.append(i)
        return res
```
# 496.下一个更大元素 I

[力扣题目链接](https://leetcode.cn/problems/next-greater-element-i/)

给你两个 没有重复元素 的数组 nums1 和 nums2 ，其中nums1 是 nums2 的子集。 请你找出 nums1 中每个元素在 nums2 中的下一个比其大的值。

```py
class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        stack = []
        hash_map = {}
        for i in nums2:
            while stack and i > stack[-1]:    #用单调栈求出nums2所有元素的下一个更大元素
                key = stack.pop()
                hash_map[key] = i        #统一存入hashmap
            stack.append(i)
        res = [-1] * len(nums1)
        for i in range(len(nums1)):
            if nums1[i] in hash_map:
                res[i] = hash_map[nums1[i]]
        return res
```
