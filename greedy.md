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

# 45.跳跃游戏 II(优解)

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

# 134. 加油站

[力扣题目链接](https://leetcode.cn/problems/gas-station/)

# 135. 分发糖果(空间优化)

[力扣题目链接](https://leetcode.cn/problems/candy/)
### Python
```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        candyVec = [1] * len(ratings)
        
        # 从前向后遍历，处理右侧比左侧评分高的情况
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i - 1]:
                candyVec[i] = candyVec[i - 1] + 1
        
        # 从后向前遍历，处理左侧比右侧评分高的情况
        for i in range(len(ratings) - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candyVec[i] = max(candyVec[i], candyVec[i + 1] + 1)
        
        # 统计结果
        result = sum(candyVec)
        return result

```
# 860.柠檬水找零

[力扣题目链接](https://leetcode.cn/problems/lemonade-change/)

# 406.根据身高重建队列(又错)

[力扣题目链接](https://leetcode.cn/problems/queue-reconstruction-by-height/)

本题有两个维度，h和k，看到这种题目一定要想如何确定一个维度，然后再按照另一个维度重新排列。

**如果两个维度一起考虑一定会顾此失彼**。

按照身高排序之后，优先按身高高的people的k来插入，后序插入节点也不会影响前面已经插入的节点，最终按照k的规则完成了队列。

```py
class Solution(object):
    def reconstructQueue(self, people):
        #先sort by reverse height，如果一样再sort by k
        people.sort(key = lambda x : (-x[0],x[1]))
        res = []
        for i in people:
            res.insert(i[1],i)
        return res
```

# 452. 用最少数量的箭引爆气球(优化)

[力扣题目链接](https://leetcode.cn/problems/minimum-number-of-arrows-to-burst-balloons/)

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if len(points) == 0: return 0
        points.sort(key=lambda x: x[0])
        result = 1
        for i in range(1, len(points)):
            if points[i][0] > points[i - 1][1]: # 气球i和气球i-1不挨着，注意这里不是>=
                result += 1     
            else:
                points[i][1] = min(points[i - 1][1], points[i][1]) # 更新重叠气球最小右边界
        return result
```

# 435. 无重叠区间

[力扣题目链接](https://leetcode.cn/problems/non-overlapping-intervals/)

# 763.划分字母区间

[力扣题目链接](https://leetcode.cn/problems/partition-labels/)

先用hashmap记录每个元素的起始和最终位置，再用类似merge interval合并。


# 56. 合并区间 (时间优化)

[力扣题目链接](https://leetcode.cn/problems/merge-intervals/)

### Python 
```python
class Solution:
    def merge(self, intervals):
        result = []
        if len(intervals) == 0:
            return result  # 区间集合为空直接返回

        intervals.sort(key=lambda x: x[0])  # 按照区间的左边界进行排序

        result.append(intervals[0])  # 第一个区间可以直接放入结果集中

        for i in range(1, len(intervals)):
            if result[-1][1] >= intervals[i][0]:  # 发现重叠区间
                # 合并区间，只需要更新结果集最后一个区间的右边界，因为根据排序，左边界已经是最小的
                result[-1][1] = max(result[-1][1], intervals[i][1])
            else:
                result.append(intervals[i])  # 区间不重叠

        return result
```

# 738.单调递增的数字

[力扣题目链接](https://leetcode.cn/problems/monotone-increasing-digits/)

题目要求小于等于N的最大单调递增的整数，那么拿一个两位的数字来举例。

例如：98，一旦出现strNum[i - 1] > strNum[i]的情况（非单调递增），首先想让strNum[i - 1]--，然后strNum[i]给为9，这样这个整数就是89，即小于98的最大的单调递增整数。

此时是从前向后遍历还是从后向前遍历呢？

从前向后遍历的话，遇到strNum[i - 1] > strNum[i]的情况，让strNum[i - 1]减一，但此时如果strNum[i - 1]减一了，可能又小于strNum[i - 2]。

举个例子，数字：332，从前向后遍历的话，那么就把变成了329，此时2又小于了第一位的3了，真正的结果应该是299。

那么从后向前遍历，就可以重复利用上次比较得出的结果了，从后向前遍历332的数值变化为：332 -> 329 -> 299


```py
class Solution(object):
    def monotoneIncreasingDigits(self, n):
        temp = list(str(n))
        length = len(temp)
        for i in range(length-1,0,-1):
            if temp[i] < temp[i-1]:
                temp[i:] = '9' * (len(temp) - i)
                temp[i-1] = str(int(temp[i-1])-1)
        return int("".join(temp))
```


# 968.监控二叉树（Hard）

[力扣题目链接](https://leetcode.cn/problems/binary-tree-cameras/)

**我们要从下往上看，局部最优：让叶子节点的父节点安摄像头，所用摄像头最少，整体最优：全部摄像头数量所用最少！**

可以使用后序遍历也就是左右中的顺序，这样就可以在回溯的过程中从下到上进行推导了。

* 0：该节点无覆盖
* 1：本节点有摄像头
* 2：本节点有覆盖

```py
class Solution(object):
    def minCameraCover(self, root):
        self.res = 0
        def dfs(root):
            if not root:
                return 2
            left = dfs(root.left)
            right = dfs(root.right)
            if left == 2 and right == 2:
                return 0
            if left == 0 or right == 0:
                self.res += 1
                return 1
            if left == 1 or right == 1:
                return 2
        top = dfs(root)
        if top == 0:
            self.res += 1        #处理单独的头部摄像头情况
        return self.res 
```

# 649. Dota2 参议院

[力扣题目链接](https://leetcode.cn/problems/dota2-senate/)

# 1221. 分割平衡字符串 

[力扣题目链接](https://leetcode.cn/problems/split-a-string-in-balanced-strings/)

