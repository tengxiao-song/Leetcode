## 什么是动态规划

所以动态规划中每一个状态一定是由上一个状态推导出来的，**这一点就区分于贪心**，贪心没有状态推导，而是从局部直接选最优的，

## 动态规划的解题步骤
**对于动态规划问题，我将拆解为如下五步曲，这五步都搞清楚了，才能说把动态规划真的掌握了！**

1. 确定dp数组（dp table）以及下标的含义
2. 确定递推公式
3. dp数组如何初始化
4. 确定遍历顺序
5. 举例推导dp数组

# 509. 斐波那契数

[力扣题目链接](https://leetcode.cn/problems/fibonacci-number/)
动规五部曲：

这里我们要用一个一维dp数组来保存递归的结果

1. 确定dp数组以及下标的含义

dp[i]的定义为：第i个数的斐波那契数值是dp[i]

2. 确定递推公式

为什么这是一道非常简单的入门题目呢？

**因为题目已经把递推公式直接给我们了：状态转移方程 dp[i] = dp[i - 1] + dp[i - 2];**

3. dp数组如何初始化

**题目中把如何初始化也直接给我们了，如下：**

```
dp[0] = 0;
dp[1] = 1;
```

4. 确定遍历顺序

从递归公式dp[i] = dp[i - 1] + dp[i - 2];中可以看出，dp[i]是依赖 dp[i - 1] 和 dp[i - 2]，那么遍历的顺序一定是从前到后遍历的

5. 举例推导dp数组

按照这个递推公式dp[i] = dp[i - 1] + dp[i - 2]，我们来推导一下，当N为10的时候，dp数组应该是如下的数列：

0 1 1 2 3 5 8 13 21 34 55

# 70. 爬楼梯

[力扣题目链接](https://leetcode.cn/problems/climbing-stairs/)

dp[i] = dp[i - 1] + dp[i - 2]

# 746. 使用最小花费爬楼梯

[力扣题目链接](https://leetcode.cn/problems/min-cost-climbing-stairs/)

# 62.不同路径

[力扣题目链接](https://leetcode.cn/problems/unique-paths/)

![image](https://github.com/tengxiao-song/Leetcode/assets/148564733/b460b7cf-65a6-4d91-bb14-2d9b9fa4e797)

# 63. 不同路径 II

[力扣题目链接](https://leetcode.cn/problems/unique-paths-ii/)

# 343. 整数拆分(难题)

[力扣题目链接](https://leetcode.cn/problems/integer-break/)

假设对正整数 i 拆分出的第一个正整数是 j（1 <= j < i），则有以下两种方案：
1) 将 i 拆分成 j 和 i−j 的和，且 i−j 不再拆分成多个正整数，此时的乘积是 j * (i-j)
2) 2) 将 i 拆分成 j 和 i−j 的和，且 i−j 继续拆分成多个正整数，此时的乘积是 j * dp[i-j]

```py
class Solution(object):
    def integerBreak(self, n):
        dp = [0] * (n+1)
        dp[2] = 1
        for i in range(3,n+1):
            for j in range(1,i//2 + 1):
                dp[i] = max(dp[i], (i-j) * j, dp[i-j]*j)
        return dp[n]
```
# 96.不同的二叉搜索树(难题)

[力扣题目链接](https://leetcode.cn/problems/unique-binary-search-trees/)

![96.不同的二叉搜索树](https://code-thinking-1253855093.file.myqcloud.com/pics/20210107093106367.png)

n为1的时候有一棵树，n为2有两棵树，这个是很直观的。

![96.不同的二叉搜索树1](https://code-thinking-1253855093.file.myqcloud.com/pics/20210107093129889.png)

来看看n为3的时候，有哪几种情况。

当1为头结点的时候，其右子树有两个节点，看这两个节点的布局，是不是和 n 为2的时候两棵树的布局是一样的啊！

当3为头结点的时候，其左子树有两个节点，看这两个节点的布局，是不是和n为2的时候两棵树的布局也是一样的啊！

当2为头结点的时候，其左右子树都只有一个节点，布局是不是和n为1的时候只有一棵树的布局也是一样的啊！

发现到这里，其实我们就找到了重叠子问题了，其实也就是发现可以通过dp[1] 和 dp[2] 来推导出来dp[3]的某种方式。

dp[3]，就是 元素1为头结点搜索树的数量 + 元素2为头结点搜索树的数量 + 元素3为头结点搜索树的数量

元素1为头结点搜索树的数量 = 右子树有2个元素的搜索树数量 * 左子树有0个元素的搜索树数量

元素2为头结点搜索树的数量 = 右子树有1个元素的搜索树数量 * 左子树有1个元素的搜索树数量

元素3为头结点搜索树的数量 = 右子树有0个元素的搜索树数量 * 左子树有2个元素的搜索树数量

有2个元素的搜索树数量就是dp[2]。

有1个元素的搜索树数量就是dp[1]。

有0个元素的搜索树数量就是dp[0]。

所以dp[3] = dp[2] * dp[0] + dp[1] * dp[1] + dp[0] * dp[2]

如图所示：

![96.不同的二叉搜索树2](https://code-thinking-1253855093.file.myqcloud.com/pics/20210107093226241.png)


此时我们已经找到递推关系了，那么可以用动规五部曲再系统分析一遍。

1. 确定dp数组（dp table）以及下标的含义

**dp[i] ： 1到i为节点组成的二叉搜索树的个数为dp[i]**。

也可以理解是i个不同元素节点组成的二叉搜索树的个数为dp[i] ，都是一样的。

以下分析如果想不清楚，就来回想一下dp[i]的定义

2. 确定递推公式

在上面的分析中，其实已经看出其递推关系， dp[i] += dp[以j为头结点左子树节点数量] * dp[以j为头结点右子树节点数量]

j相当于是头结点的元素，从1遍历到i为止。

所以递推公式：dp[i] += dp[j - 1] * dp[i - j]; ，j-1 为j为头结点左子树节点数量，i-j 为以j为头结点右子树节点数量

3. dp数组如何初始化

初始化，只需要初始化dp[0]就可以了，推导的基础，都是dp[0]。

从定义上来讲，空节点也是一棵二叉树，也是一棵二叉搜索树, 所以初始化dp[0] = 1 

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0] * (n + 1)  # 创建一个长度为n+1的数组，初始化为0
        dp[0] = 1  # 当n为0时，只有一种情况，即空树，所以dp[0] = 1
        for i in range(1, n + 1):  # 遍历从1到n的每个数字
            for j in range(1, i + 1):  # 对于每个数字i，计算以i为根节点的二叉搜索树的数量
                dp[i] += dp[j - 1] * dp[i - j]  # 利用动态规划的思想，累加左子树和右子树的组合数量
        return dp[n]  # 返回以1到n为节点的二叉搜索树的总数量
```

# 背包

有n件物品和一个最多能背重量为w 的背包。第i件物品的重量是weight[i]，得到的价值是value[i] 。**每件物品只能用一次**，求解将哪些物品装入背包里物品价值总和最大。

这样其实是没有从底向上去思考，而是习惯性想到了背包，那么暴力的解法应该是怎么样的呢？

每一件物品其实只有两个状态，取或者不取，所以可以使用回溯法搜索出所有的情况，那么时间复杂度就是$o(2^n)$，这里的n表示物品数量。

**所以暴力的解法是指数级别的时间复杂度。进而才需要动态规划的解法来进行优化！**

在下面的讲解中，我举一个例子：

背包最大重量为4。

物品为：

|       | 重量 | 价值 |
| ----- | ---- | ---- |
| 物品0 | 1    | 15   |
| 物品1 | 3    | 20   |
| 物品2 | 4    | 30   |

问背包能背的物品最大价值是多少？

### 二维dp数组01背包

依然动规五部曲分析一波。

1. 确定dp数组以及下标的含义

对于背包问题，有一种写法， 是使用二维数组，即**dp[i][j] 表示从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少**。

只看这个二维数组的定义，大家一定会有点懵，看下面这个图：

![动态规划-背包问题1](https://code-thinking-1253855093.file.myqcloud.com/pics/20210110103003361.png)

**要时刻记着这个dp数组的含义，下面的一些步骤都围绕这dp数组的含义进行的**，如果哪里看懵了，就来回顾一下i代表什么，j又代表什么。

2. 确定递推公式

再回顾一下dp[i][j]的含义：从下标为[0-i]的物品里任意取，放进容量为j的背包，价值总和最大是多少。

那么可以有两个方向推出来dp[i][j]，

* **不放物品i**：由dp[i - 1][j]推出，即背包容量为j，里面不放物品i的最大价值，此时dp[i][j]就是dp[i - 1][j]。(其实就是当物品i的重量大于背包j的重量时，物品i无法放进背包中，所以背包内的价值依然和前面相同。)
* **放物品i**：由dp[i - 1][j - weight[i]]推出，dp[i - 1][j - weight[i]] 为背包容量为j - weight[i]的时候不放物品i的最大价值，那么dp[i - 1][j - weight[i]] + value[i] （物品i的价值），就是背包放物品i得到的最大价值

所以递归公式： dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]);

3. dp数组如何初始化

**关于初始化，一定要和dp数组的定义吻合，否则到递推公式的时候就会越来越乱**。

首先从dp[i][j]的定义出发，如果背包容量j为0的话，即dp[i][0]，无论是选取哪些物品，背包价值总和一定为0。如图：

![动态规划-背包问题2](https://code-thinking-1253855093.file.myqcloud.com/pics/2021011010304192.png)

在看其他情况。

状态转移方程 dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]); 可以看出i 是由 i-1 推导出来，那么i为0的时候就一定要初始化。

dp[0][j]，即：i为0，存放编号0的物品的时候，各个容量的背包所能存放的最大价值。

那么很明显当 j < weight[0]的时候，dp[0][j] 应该是 0，因为背包容量比编号0的物品重量还小。

当j >= weight[0]时，dp[0][j] 应该是value[0]，因为背包容量放足够放编号0物品。

此时dp数组初始化情况如图所示：

![动态规划-背包问题7](https://code-thinking-1253855093.file.myqcloud.com/pics/20210110103109140.png)

dp[0][j] 和 dp[i][0] 都已经初始化了，那么其他下标应该初始化多少呢？

其实从递归公式： dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]); 可以看出dp[i][j]  是由左上方数值推导出来了，那么 其他下标初始为什么数值都可以，因为都会被覆盖。

**初始-1，初始-2，初始100，都可以！**

但只不过一开始就统一把dp数组统一初始为0，更方便一些。

如图：

![动态规划-背包问题10](https://code-thinking-1253855093.file.myqcloud.com/pics/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92-%E8%83%8C%E5%8C%85%E9%97%AE%E9%A2%9810.jpg)

**费了这么大的功夫，才把如何初始化讲清楚，相信不少同学平时初始化dp数组是凭感觉来的，但有时候感觉是不靠谱的**。

4. 确定遍历顺序


在如下图中，可以看出，有两个遍历的维度：物品与背包重量

![动态规划-背包问题3](https://code-thinking-1253855093.file.myqcloud.com/pics/2021011010314055.png)

那么问题来了，**先遍历 物品还是先遍历背包重量呢？**

**其实都可以！！ 但是先遍历物品更好理解**。

**虽然两个遍历的次序不同，但是dp[i][j]所需要的数据就是左上角，根本不影响dp[i][j]公式的推导！**

但先遍历物品再遍历背包这个顺序更好理解。

**其实背包问题里，两个for循环的先后循序是非常有讲究的，理解遍历顺序其实比理解推导公式难多了**。

5. 举例推导dp数组

来看一下对应的dp数组的数值，如图：

![动态规划-背包问题4](https://code-thinking-1253855093.file.myqcloud.com/pics/20210118163425129.jpg)

最终结果就是dp[2][4]。

建议大家此时自己在纸上推导一遍，看看dp数组里每一个数值是不是这样的。

**做动态规划的题目，最好的过程就是自己在纸上举一个例子把对应的dp数组的数值推导一下，然后在动手写代码！**

很多同学做dp题目，遇到各种问题，然后凭感觉东改改西改改，怎么改都不对，或者稀里糊涂就改过了。

主要就是自己没有动手推导一下dp数组的演变过程，如果推导明白了，代码写出来就算有问题，只要把dp数组打印出来，对比一下和自己推导的有什么差异，很快就可以发现问题了。

```python
def test_2_wei_bag_problem1(weight, value, bagweight):
    dp = [[0] * (bagweight + 1) for _ in range(len(weight))]

    # 初始化
    for j in range(weight[0], bagweight + 1):
        dp[0][j] = value[0]

    # weight数组的大小就是物品个数
    for i in range(1, len(weight)):  # 遍历物品
        for j in range(bagweight + 1):  # 遍历背包容量
            if j < weight[i]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])

    return dp[len(weight) - 1][bagweight]
```

# 一维滚动数组背包

必须是先遍历物品在外，容量在里。而且要用后续遍历保证只添加每个物品一次。

```python
def test_1_wei_bag_problem():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bagWeight = 4
    # 初始化
    dp = [0] * (bagWeight + 1)
    for i in range(len(weight)):  # 遍历物品
        for j in range(bagWeight, weight[i] - 1, -1):  # 遍历背包容量， 后续遍历保证只添加每个物品一次
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    print(dp[bagWeight])
```

# 416. 分割等和子集

[力扣题目链接](https://leetcode.cn/problems/partition-equal-subset-sum/)

这道题目是要找是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

那么只要找到集合里能够出现 sum / 2 的子集总和，就算是可以分割成两个相同元素和子集了。相当于填满容量为sum / 2的背包。

本题中每一个元素的数值既是重量，也是价值。

**此时问题就转化为，能否装满容量为x的背包**。

```py
class Solution(object):
    def canPartition(self, nums):
        if sum(nums) % 2 == 1:
            return False
        mid = sum(nums) / 2
        dp = [0] * (mid+1)
        for i in nums:
            for j in range(mid, i-1,-1):
                dp[j] = max(dp[j], dp[j-i] + i)
                if dp[j] == mid:
                    return True
        return False
```

# 1049.最后一块石头的重量II

[力扣题目链接](https://leetcode.cn/problems/last-stone-weight-ii/)

本题其实就是尽量让石头分成重量相同的两堆，相撞之后剩下的石头最小，**这样就化解成01背包问题了**。

**此时问题就转化为，尽量装满容量为x的背包**。

```py
class Solution(object):
    def lastStoneWeightII(self, stones):
        total = sum(stones)
        target = total / 2        #尽量去凑一半的重量，这样对撞就是0
        dp = [0] * (target + 1)    #index 0代表容量0
        for i in stones:
            for j in range(target, i-1, -1):    #终止条件为i-1，因为之后容量放不下物品i
                dp[j] = max(dp[j], dp[j-i] + i)
        return total - dp[target] - dp[target]    #用另一半石头减去这一半石头
```

# 494.目标和

[力扣题目链接](https://leetcode.cn/problems/target-sum/)

本题要如何使表达式结果为target，

既然为target，那么就一定有 left组合 - right组合 = target。

left + right = sum，而sum是固定的。right = sum - left  

公式来了， left - (sum - left) = target 推导出  left = (target + sum)/2 。

target是固定的，sum是固定的，left就可以求出来。

此时问题就是在集合nums中找出和为left的组合。

**此时问题就转化为，装满容量为x的背包，有几种方法**。

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)  # 计算nums的总和
        if abs(target) > total_sum:
            return 0  # 此时没有方案
        if (target + total_sum) % 2 == 1:
            return 0  # 此时没有方案
        target_sum = (target + total_sum) // 2  # 目标和
        dp = [0] * (target_sum + 1)  # 创建动态规划数组，初始化为0
        dp[0] = 1  # 当目标和为0时，只有一种方案，即什么都不选
        for num in nums:
            for j in range(target_sum, num - 1, -1):
                dp[j] += dp[j - num]  # 状态转移方程，累加不同选择方式的数量
        return dp[target_sum]  # 返回达到目标和的方案数

```

# 474.一和零

[力扣题目链接](https://leetcode.cn/problems/ones-and-zeroes/)

这个背包有两个维度，一个是m 一个是n，而不同长度的字符串就是不同大小的待装物品。

**此时问题就转化为，装满容量为x,y的背包的最大个数**。

1. 确定dp数组（dp table）以及下标的含义

**dp[i][j]：最多有i个0和j个1的strs的最大子集的大小为dp[i][j]**。

2. 确定递推公式

dp[i][j] 可以由前一个strs里的字符串推导出来，strs里的字符串有zeroNum个0，oneNum个1。

dp[i][j] 就可以是 dp[i - zeroNum][j - oneNum] + 1。

然后我们在遍历的过程中，取dp[i][j]的最大值。

所以递推公式：dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1);

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]  # 创建二维动态规划数组，初始化为0
        for s in strs:  # 遍历物品
            zeroNum = s.count('0')  # 统计0的个数
            oneNum = len(s) - zeroNum  # 统计1的个数
            for i in range(m, zeroNum - 1, -1):  # 遍历背包容量且从后向前遍历
                for j in range(n, oneNum - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1)  # 状态转移方程
        return dp[m][n]
```


# 动态规划：完全背包理论基础

有N件物品和一个最多能背重量为W的背包。第i件物品的重量是weight[i]，得到的价值是value[i] 。**每件物品都有无限个（也就是可以放入背包多次）**，求解将哪些物品装入背包里物品价值总和最大。

**完全背包和01背包问题唯一不同的地方就是，每种物品有无限件**。

在下面的讲解中，我依然举这个例子：

背包最大重量为4。

物品为：

|       | 重量 | 价值 |
| ---   | ---  | ---  |
| 物品0 | 1    | 15   |
| 物品1 | 3    | 20   |
| 物品2 | 4    | 30   |

**每件商品都有无限个！**

问背包能背的物品最大价值是多少？

01背包和完全背包唯一不同就是体现在遍历顺序上，所以本文就不去做动规五部曲了，我们直接针对遍历顺序经行分析！

```CPP
// 先遍历物品，再遍历背包
for(int i = 0; i < weight.size(); i++) { // 遍历物品
    for(int j = weight[i]; j <= bagWeight ; j++) { // 遍历背包容量
        dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);

    }
}
```

**同样，在完全背包中，对于一维dp数组来说，其实两个for循环嵌套顺序是无所谓的！**

# 518.零钱兑换II

[力扣题目链接](https://leetcode.cn/problems/coin-change-ii/)

给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。 

求装满重量为amount的背包有多少种方法，每个物品可以添加无限次。

```py
class Solution(object):
    def change(self, amount, coins):
        dp = [0] * (amount + 1)
        dp[0] = 1
        for i in coins:
            for j in range(i, amount+1):
                dp[j] += dp[j-i]
        return dp[amount]
```

在求装满背包有几种方案的时候，认清遍历顺序是非常关键的。

**如果求组合数就是外层for循环遍历物品，内层for遍历背包**。

**如果求排列数就是外层for遍历背包，内层for循环遍历物品**。

# 377. 组合总和 Ⅳ

[力扣题目链接](https://leetcode.cn/problems/combination-sum-iv/)

给定一个由正整数组成且不存在重复数字的数组，找出和为给定目标正整数的组合的个数。 这里顺序不同的序列被视作不同的组合。

个数可以不限使用，说明这是一个完全背包。

得到的集合是排列，说明需要考虑元素之间的顺序。

**如果求排列数就是外层for遍历背包，内层for循环遍历物品**。

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(1, target + 1):  # 遍历背包
            for j in range(len(nums)):  # 遍历物品
                if i - nums[j] >= 0:
                    dp[i] += dp[i - nums[j]]
        return dp[target]
```

# 322. 零钱兑换

[力扣题目链接](https://leetcode.cn/problems/coin-change/)

给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

你可以认为每种硬币的数量是无限的。**求装满背包最小物品个数**

首先凑足总金额为0所需钱币的个数一定是0，那么dp[0] = 0;

考虑到递推公式的特性，dp[j]必须初始化为一个最大的数，否则就会在min(dp[j - coins[i]] + 1, dp[j])比较的过程中被初始值覆盖。

所以下标非0的元素都是应该是最大值。

本题求钱币最小个数，**那么钱币有顺序和没有顺序都可以，都不影响钱币的最小个数**。所以本题并不强调集合是组合还是排列。

```py
class Solution(object):
    def coinChange(self, coins, amount):
        dp = [float("inf")] * (amount + 1)
        dp[0] = 0
        for i in coins:
            for j in range(i,amount+1):
                dp[j] = min(dp[j],dp[j-i]+1)
        return dp[amount] if dp[amount] != float("inf") else -1
```

# 279.完全平方数

[力扣题目链接](https://leetcode.cn/problems/perfect-squares/)

**我来把题目翻译一下：完全平方数就是物品（可以无限件使用），凑个正整数n就是背包，问凑满这个背包最少有多少物品？**

dp[j] 可以由dp[j - i * i]推出， dp[j - i * i] + 1 便可以凑成dp[j]。

此时我们要选择最小的dp[j]，所以递推公式：dp[j] = min(dp[j - i * i] + 1, dp[j]);

```py
class Solution(object):
    def numSquares(self, n):
        dp = [float("inf")] * (n+1)
        dp[0] = 0
        dp[1] = 1
        for i in range(1,n/2+1):    # 遍历物品
            for j in range(i*i,n+1):    # 遍历背包,直接从i*i开始
                dp[j] = min(dp[j],dp[j-i*i]+1)
        return dp[n]
```

# 139.单词拆分

[力扣题目链接](https://leetcode.cn/problems/word-break/)

单词就是物品，字符串s就是背包，单词能否组成字符串s，就是问物品能不能把背包装满。

递推公式：**dp[i] : 字符串长度为i的话，dp[i]为true，表示可以拆分为一个或多个在字典中出现的单词**。如果确定dp[j] 是true，且 [j, i] 这个区间的子串出现在字典里，那么dp[i]一定是true。（j < i ）。

所以递推公式是 if([j, i] 这个区间的子串出现在字典里 && dp[j]是true) 那么 dp[i] = true。

"apple", "pen" 是物品，那么我们要求 物品的组合一定是 "apple" + "pen" + "apple" 才能组成 "applepenapple"。  所以说，本题一定是 先遍历 背包，再遍历物品。 

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordSet = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)  # dp[i] 表示字符串的前 i 个字符是否可以被拆分成单词
        dp[0] = True  # 初始状态，空字符串可以被拆分成单词

        for i in range(1, n + 1): # 遍历背包
            for j in range(i): # 遍历单词,取j到i的substr
                if dp[j] and s[j:i] in wordSet:
                    dp[i] = True  # 如果 s[0:j] 可以被拆分成单词，并且 s[j:i] 在单词集合中存在，则 s[0:i] 可以被拆分成单词
                    break
        return dp[n]
```

# 198.打家劫舍

[力扣题目链接](https://leetcode.cn/problems/house-robber/)

```
class Solution(object):
    def rob(self, nums):
        length = len(nums)
        dp = [0] * (length + 1)
        dp[1] = nums[0]
        for i in range(2,length+1):
            dp[i] = max(dp[i-2]+nums[i-1],dp[i-1])    # 对于每个房屋，选择抢劫当前房屋和抢劫前一个房屋的最大金额
        return dp[length]
```

# 213.打家劫舍II

[力扣题目链接](https://leetcode.cn/problems/house-robber-ii/)

* 情况一：考虑包含首元素，不包含尾元素

![213.打家劫舍II1](https://code-thinking-1253855093.file.myqcloud.com/pics/20210129160821374-20230310134003961.jpg)

* 情况二：考虑包含尾元素，不包含首元素

![213.打家劫舍II2](https://code-thinking-1253855093.file.myqcloud.com/pics/20210129160842491-20230310134008133.jpg)

```py
class Solution(object):
    def rob(self, nums):
        length = len(nums)
        if length < 3:
            return max(nums)
        dp = [0] * (length)
        dp1 = [0] * (length)
        num1 = nums[1:]
        dp[1] = nums[0]
        dp1[1] = num1[0]
        for i in range(2,length):
            dp[i] = max(dp[i-1], dp[i-2]+nums[i-1])
        for i in range(2,length):
            dp1[i] = max(dp1[i-1], dp1[i-2]+num1[i-1])
        return max(dp[length-1],dp1[length-1])
```

# 337.打家劫舍 III

[力扣题目链接](https://leetcode.cn/problems/house-robber-iii/)

dp数组（dp table）以及下标的含义：下标为0记录不偷该节点所得到的的最大金钱，下标为1记录偷该节点所得到的的最大金钱。

在遍历的过程中，如果遇到空节点的话，很明显，无论偷还是不偷都是0，所以就返回[0, 0]

使用后序遍历。 因为要通过递归函数的返回值来做下一步计算。

如果是偷当前节点，那么左右孩子就不能偷，val1 = cur->val + left[0] + right[0];  （**如果对下标含义不理解就再回顾一下dp数组的含义**）

如果不偷当前节点，那么左右孩子就可以偷，至于到底偷不偷一定是选一个最大的，所以：val2 = max(left[0], left[1]) + max(right[0], right[1]);

最后当前节点的状态就是{val2, val1}; 即：{不偷当前节点得到的最大金钱，偷当前节点得到的最大金钱}

```py
class Solution(object):
    def rob(self, root):
        def dfs(root):
            if not root:
                return [0,0]
            dpleft = dfs(root.left)
            dpright = dfs(root.right)
            robval = root.val + dpleft[0] + dpright[0]
            norobval = max(dpleft[0],dpleft[1]) + max(dpright[0],dpright[1])
            return [norobval, robval]
        dp = dfs(root)
        return max(dp)
```

# 121. 买卖股票的最佳时机

[力扣题目链接](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

dp[i][0] 表示第i天持有股票所得最多现金; dp[i][1] 表示第i天不持有股票所得最多现金.

如果第i天持有股票即dp[i][0]， 那么可以由两个状态推出来
* 第i-1天就持有股票，那么就保持现状，所得现金就是昨天持有股票的所得现金 即：dp[i - 1][0]
* 第i天买入股票，所得现金就是买入今天的股票后所得现金即：-prices[i]

那么dp[i][0]应该选所得现金最大的，所以dp[i][0] = max(dp[i - 1][0], -prices[i]);

如果第i天不持有股票即dp[i][1]， 也可以由两个状态推出来
* 第i-1天就不持有股票，那么就保持现状，所得现金就是昨天不持有股票的所得现金 即：dp[i - 1][1]
* 第i天卖出股票，所得现金就是按照今天股票价格卖出后所得现金即：prices[i] + dp[i - 1][0]

同样dp[i][1]取最大的，dp[i][1] = max(dp[i - 1][1], prices[i] + dp[i - 1][0]);

其基础都是要从dp[0][0]和dp[0][1]推导出来。

那么dp[0][0]表示第0天持有股票，此时的持有股票就一定是买入股票了，因为不可能有前一天推出来，所以dp[0][0] -= prices[0];

dp[0][1]表示第0天不持有股票，不持有股票那么现金就是0，所以dp[0][1] = 0;

```py
class Solution(object):
    def maxProfit(self, prices):
        length = len(prices)
        dp = [[0]*2 for _ in range(length)]
        dp[0][0] = -prices[0]
        for i in range(1,length):
            dp[i][0] = max(dp[i-1][0], -prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
        return max(dp[length-1][0],dp[length-1][1])
```

# 122.买卖股票的最佳时机II 

[力扣题目链接](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

本题因为一只股票可以买卖多次，所以当第i天买入股票的时候，所持有的现金可能有之前买卖过的利润。

那么第i天持有股票即dp[i][0]，如果是第i天买入股票，所得现金就是昨天不持有股票的所得现金 减去 今天的股票价格 即：dp[i - 1][1] - prices[i]。

```py
class Solution(object):
    def maxProfit(self, prices):
        length = len(prices)
        dp = [[0]*2 for _ in range(length)]
        dp[0][0] = -prices[0]
        for i in range(1, length):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
        return max(dp[length-1][0], dp[length-1][1])
```

# 123.买卖股票的最佳时机III

[力扣题目链接](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/)

设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。

一天一共就有四个状态，

1. 第一次持有股票 (第一次持有股票并不是说一定要第i天买入股票，可以延续前一天第一次持有股票的状态)
2. 第一次不持有股票 
3. 第二次持有股票
4. 第二次不持有股票

```py
class Solution(object):
    def maxProfit(self, prices):
        length = len(prices)
        dp = [[0]*4 for _ in range(length)] #每天四个状态
        dp[0][0] = -prices[0]    #最初的第一次买入
        dp[0][2] = -prices[0]    #最初的第二次买入,当天买卖再买
        for i in range(1,length):
            dp[i][0] = max(dp[i-1][0], -prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] + prices[i])
            dp[i][2] = max(dp[i-1][2], dp[i-1][1] - prices[i])
            dp[i][3] = max(dp[i-1][3], dp[i-1][2] + prices[i])
        return dp[length-1][3]    #第二次卖出手里所剩的钱一定是最多的, 如果第一次卖出已经是最大值了，那么我们可以在当天立刻买入再立刻卖出。所以dp[4][4]已经包含了dp[4][2]的情况
```

# 188.买卖股票的最佳时机IV

[力扣题目链接](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/)

这里要求至多有k次交易. 每次交易都会多出两个状态,例如k=2, 有第一次持有，不持有和第二次持有，不持有。所以每天dp数组有2k个状态。

```py
class Solution(object):
    def maxProfit(self, k, prices):
        length = len(prices)
        dp = [[0]*(2*k) for _ in range(length)]
        for i in range(0,2*k,2):
            dp[0][i] = -prices[0]    #初始化所有第i次持有
        for i in range(1,length):
            for j in range(2*k):
                if j == 0:
                    dp[i][j] = max(dp[i-1][j], -prices[i])    #第一次持有必定是从零开始买进或者延续昨天
                elif j % 2 == 1:
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j-1] + prices[i])    #奇数必定是第j次不持有，所以延续或者卖掉赚钱
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i-1][j-1] - prices[i])    #偶数必定是第j次持有，所以延续或者买进
        return dp[length-1][2*k-1]
```

# 309.最佳买卖股票时机含冷冻期

[力扣题目链接](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

* 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

具体可以区分出如下四个状态：

* 状态一：持有股票状态（之前就买入了股票然后没有操作，一直持有, 或者是今天买入股票，）
    * 操作一：前一天就是持有股票状态（状态一），dp[i][0] = dp[i - 1][0]
    * 操作二：今天买入了，有两种情况
        * 前一天是冷冻期（状态四），dp[i - 1][3] - prices[i]
        * 前一天是保持卖出股票的状态（状态二），dp[i - 1][1] - prices[i]
* 不持有股票状态，这里就有两种卖出股票状态
    * 状态二：保持卖出股票的状态（前一天就是卖出股票状态，一直没操作, 或者是两天前卖出了股票度过一天冷冻期）
      * 操作一：前一天就是状态二, dp[i - 1][1]
      * 操作二：前一天是冷冻期（状态四), dp[i - 1][3]
    * 状态三：今天卖出股票
        * 只有一个操作：昨天一定是持有股票状态（状态一），今天卖出
* 状态四：今天为冷冻期状态，但冷冻期状态不可持续，只有一天！
    * 只有一个操作：昨天卖出了股票（状态三）

 ```CPP
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if (n == 0) return 0;
        vector<vector<int>> dp(n, vector<int>(4, 0));
        dp[0][0] -= prices[0]; // 持股票
        for (int i = 1; i < n; i++) {
            dp[i][0] = max(dp[i - 1][0], max(dp[i - 1][3] - prices[i], dp[i - 1][1] - prices[i]));
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][3]);
            dp[i][2] = dp[i - 1][0] + prices[i];
            dp[i][3] = dp[i - 1][2];
        }
        return max(dp[n - 1][3], max(dp[n - 1][1], dp[n - 1][2]));
    }
};
```
# 714.买卖股票的最佳时机含手续费

[力扣题目链接](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

你可以无限次地完成交易，但是你每笔交易都需要付手续费。再卖出时减去手续费即可。


# 300.最长递增子序列

[力扣题目链接](https://leetcode.cn/problems/longest-increasing-subsequence/)

给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。

**dp[i]表示i之前包括i的以nums[i]结尾的最长递增子序列的长度** 

```py
class Solution(object):
    def lengthOfLIS(self, nums):
        length = len(nums)
        dp = [1]*length    #所有元素自己都有一长度
        for i in range(1,length):
            for j in range(i):        #不要求连续，所以可以从任意子序列加当前元素
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i],dp[j]+1)
        return max(dp)        #最大子序列不一定以最后一个元素结尾，需要返回整个dp最大的一个
```

# 674. 最长连续递增序列

[力扣题目链接](https://leetcode.cn/problems/longest-continuous-increasing-subsequence/)

给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。

因为本题要求连续递增子序列，所以就只要比较nums[i]与nums[i - 1].
