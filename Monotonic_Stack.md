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
