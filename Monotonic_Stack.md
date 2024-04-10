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

# 503.下一个更大元素II

[力扣题目链接](https://leetcode.cn/problems/next-greater-element-ii/)

给定一个循环数组（最后一个元素的下一个元素是数组的第一个元素），输出每个元素的下一个更大元素。数字 x 的下一个更大的元素是按数组遍历顺序，这个数字之后的第一个比它更大的数，这意味着你应该循环地搜索它的下一个更大的数。如果不存在，则输出 -1。

方法1:两个原数组拼接在一起。方法2:用mod，省空间。
```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        dp = [-1] * len(nums)
        stack = []
        for i in range(len(nums)*2):
            while(len(stack) != 0 and nums[i%len(nums)] > nums[stack[-1]]):
                    dp[stack[-1]] = nums[i%len(nums)]
                    stack.pop()
            stack.append(i%len(nums))
        return dp
```

# 42. 接雨水

[力扣题目链接](https://leetcode.cn/problems/trapping-rain-water/)


首先单调栈是按照行方向来计算雨水.

![42.接雨水2](https://code-thinking-1253855093.file.myqcloud.com/pics/20210223092629946.png)

从栈头（元素从栈头弹出）到栈底的顺序应该是从小到大的顺序。

因为一旦发现添加的柱子高度大于栈头元素了，此时就出现凹槽了，栈头元素就是凹槽底部的柱子，栈头第二个元素就是凹槽左边的柱子，而添加的元素就是凹槽右边的柱子。

栈里就存放下标

```py
class Solution:
    def trap(self, height: List[int]) -> int:
        stack = [0]
        result = 0
        for i in range(1, len(height)):
            while stack and height[i] > height[stack[-1]]:
                mid_height = stack.pop()
                if stack:
                    # 雨水高度是 min(凹槽左侧高度, 凹槽右侧高度) - 凹槽底部高度
                    h = min(height[stack[-1]], height[i]) - height[mid_height]
                    # 雨水宽度是 凹槽右侧的下标 - 凹槽左侧的下标 - 1
                    w = i - stack[-1] - 1
                    # 累计总雨水体积
                    result += h * w
            stack.append(i)
        return result
```
