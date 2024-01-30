# 455.分发饼干

[力扣题目链接](https://leetcode.cn/problems/assign-cookies/)

# 376. 摆动序列(优解)

[力扣题目链接](https://leetcode.cn/problems/wiggle-subsequence/)
```python
class Solution:
    def wiggleMaxLength(self, nums):
        if len(nums) <= 1:
            return len(nums)  # 如果数组长度为0或1，则返回数组长度
        curDiff = 0  # 当前一对元素的差值
        preDiff = 0  # 前一对元素的差值
        result = 1  # 记录峰值的个数，初始为1（默认最右边的元素被视为峰值）
        for i in range(len(nums) - 1):
            curDiff = nums[i + 1] - nums[i]  # 计算下一个元素与当前元素的差值
            # 如果遇到一个峰值
            if (preDiff <= 0 and curDiff > 0) or (preDiff >= 0 and curDiff < 0):
                result += 1  # 峰值个数加1
                preDiff = curDiff  # 注意这里，只在摆动变化的时候更新preDiff
        return result  # 返回最长摆动子序列的长度
```

# 53. 最大子序和

[力扣题目链接](https://leetcode.cn/problems/maximum-subarray/)

# 122.买卖股票的最佳时机 II

[力扣题目链接](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

# 55. 跳跃游戏

[力扣题目链接](https://leetcode.cn/problems/jump-game/)

# 45.跳跃游戏 II(又错)

[力扣题目链接](https://leetcode.cn/problems/jump-game-ii/)

```py
class Solution(object):
    def jump(self, nums):
        res, reach, maxreach = 0, 0, 0
        for i in range(len(nums)-1):    //不用检查最后一位
            maxreach = max(maxreach,i + nums[i])
            if i == reach:
                res += 1
                reach = maxreach
        return res
```

# 1005.K次取反后最大化的数组和(错)

[力扣题目链接](https://leetcode.cn/problems/maximize-sum-of-array-after-k-negations/)

```py
class Solution(object):
    def largestSumAfterKNegations(self, nums, k):
        nums.sort(key=lambda x: abs(x), reverse=True)    #按照绝对值大小降序排序
        for i in range(len(nums)):
            if nums[i] < 0 and k > 0:    #只在当前元素是负以及还有k翻转
                nums[i] *= -1
                k -= 1
        if k % 2 == 1:        #当k为奇数，翻转影响最小的元素
            nums[-1] *= -1
        return sum(nums)
```
