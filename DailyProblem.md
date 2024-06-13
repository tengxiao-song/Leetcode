# 705 Design Hashset

[力扣题目链接](https://leetcode.cn/problems/design-hashset/description/)

### 方法1: array

创建一个巨大的array，然后每次add，remove和contains都对key对应的下标进行处理。时间O(1), 空间浪费。

### 方法2: close hashing

时间复杂度：O(n/b), 其中 n 为哈希表中的元素数量，b 为链表的数量。空间复杂度：O(n+b)

```py
class MyHashSet(object):

    def __init__(self):
        self.base = 769
        self.nums = [[]]*self.base

    def add(self, key):
        """
        :type key: int
        :rtype: None
        """
        new_key = key % self.base
        val = self.nums[new_key]
        if key not in val:
            val.insert(0,key)


    def remove(self, key):
        """
        :type key: int
        :rtype: None
        """
        new_key = key % self.base
        val = self.nums[new_key]
        if key in val:
            val.remove(key)


    def contains(self, key):
        """
        :type key: int
        :rtype: bool
        """
        new_key = key % self.base
        val = self.nums[new_key]
        return key in val
```

# 706 Design HashTable

[力扣题目链接](https://leetcode.cn/problems/design-hashmap/)

「设计哈希映射」与「设计哈希集合」解法接近，唯一的区别在于我们存储的不是 key 本身，而是 (key,value)。

```py
class MyHashMap(object):

    def __init__(self):
        self.magic_num = 1009
        self.res = [[]]*self.magic_num

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        index = key % self.magic_num
        for i, (k,v) in enumerate(self.res[index]):    #use enumerate to modify, or else only able to modify copies of k,v
            if k == key:
                self.res[index][i] = (key,value)
                return
        self.res[index].append((key,value))

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        index = key % self.magic_num
        for k, v in self.res[index]:
            if k == key:
                return v
        return -1

    def remove(self, key):
        """
        :type key: int
        :rtype: None
        """
        index = key % self.magic_num
        for i, (k, v) in enumerate(self.res[index]):
            if k == key:
                self.res[index].pop(i)
                return
```

# 623. Add One Row to Tree

[力扣题目链接](https://leetcode.cn/problems/add-one-row-to-tree/description/)

# 988. Smallest String Starting From Leaf

[力扣题目链接](https://leetcode.cn/problems/smallest-string-starting-from-leaf/)

用dfs暴力递归所有root到leaf的string，保留最小的。

```py
class Solution(object):
    def smallestFromLeaf(self, root):
        self.res = "~"    # ~是最大的字符，会被替代
        def dfs(root, A):
            if not root:
                return
            A.append(chr(root.val+97))
            if not root.left and not root.right:
                self.res = min(self.res, "".join(reversed(A)))
            dfs(root.left, A)
            dfs(root.right, A)
            A.pop()
        dfs(root, [])
        return self.res
```

# 1883. Minimum Skips to Arrive at Meeting On Time(困难，精度还有问题)

[力扣题目链接](https://leetcode.cn/problems/minimum-skips-to-arrive-at-meeting-on-time/)

dp定义: dp[i][j] 表示经过了 dist[0]到dist[i−1]的i段道路，并且跳过了j次的最短用时。

如果跳过dp[i][j] = ceil(dp[i - 1][j] + dist[i - 1] / speed),
如果不跳过dp[i][j] = dp[i - 1][j - 1] + dist[i - 1] / speed)

初始化dp[0][0] = 0

```py
class Solution:
    def minSkips(self, dist: List[int], speed: int, hoursBefore: int) -> int:
        # 可忽略误差
        EPS = 1e-7
        n = len(dist)
        f = [[float("inf")] * (n + 1) for _ in range(n + 1)]
        f[0][0] = 0.
        for i in range(1, n + 1):
            for j in range(i + 1):
                if j != i:
                    f[i][j] = min(f[i][j], ceil(f[i - 1][j] + dist[i - 1] / speed - EPS))
                if j != 0:
                    f[i][j] = min(f[i][j], f[i - 1][j - 1] + dist[i - 1] / speed)
        
        for j in range(n + 1):
            if f[n][j] < hoursBefore + EPS:
                return j    #最后返回可以准时抵达的最小跳过次数
        return -1
```

# 216. Combination Sum III

[力扣题目链接](https://leetcode.cn/problems/combination-sum-iii/)

经典回溯模版

```py
class Solution(object):
    def combinationSum3(self, k, n):
        self.res = []
        def dfs(temp, n, k, num):
            if sum(temp) == n and len(temp) == k:
                self.res.append(temp[:])
                return
            elif sum(temp) < n and len(temp) < k:
                for i in range(num, 10):
                    temp.append(i)
                    dfs(temp, n, k,i+1)
                    temp.pop()    #剪枝
        dfs([],n, k, 1)
        return self.res
```

# 1052. Grumpy Bookstore Owner

[力扣题目链接](https://leetcode.cn/problems/grumpy-bookstore-owner/description/)


# 2385. Amount of Time for Binary Tree to Be Infected

[力扣题目链接](https://leetcode.cn/problems/amount-of-time-for-binary-tree-to-be-infected/description/)

用dfs和hashtable建graph，用bfs扩散

```py
class Solution(object):
    def amountOfTime(self, root, start):
        dic = collections.defaultdict(list)
        def dfs(root):
            if not root:
                return
            if root.left:
                dic[root.val].append(root.left.val)
                dic[root.left.val].append(root.val)
            if root.right:
                dic[root.val].append(root.right.val)
                dic[root.right.val].append(root.val)
            dfs(root.left)
            dfs(root.right)
        dfs(root)

        minute = defaultdict(int)
        minute[start] = 0
        res = 0
        q = collections.deque([start])
        while q:
            u = q.popleft()
            for v in dic[u]:
                if v not in minute:
                    q.append(v)
                    minute[v] = minute[u] + 1
                    res = max(res, minute[v])
        return res

```

# 1017. Convert to Base -2

[力扣题目链接](https://leetcode.cn/problems/convert-to-base-2/description)

类似于2进制的运算逻辑:
1. 获取余数
2. 将余数加入答案
3. n减去余数，除以进制

其实2进制也需要n减去余数，但是可以忽略, 因为n一定是正的而且py除法自动向下取整了。

但是-2进制，n有可能是负的，所以一定要减去余数。
```py
class Solution(object):
    def baseNeg2(self, n):
        if n == 0: return "0"
        res = []
        while n != 0:
            remain = abs(n % -2)
            if remain == 0:
                res.append("0")
            else:
                res.append("1")
            n = (n - remain) / (-2)
        return "".join(res[::-1])    #余数从后往前就是转化后的数，这里用list的append所以要翻转

```

# 1329. Sort the Matrix Diagonally

[力扣题目链接](https://leetcode.cn/problems/sort-the-matrix-diagonally/description)

1. 首先要分析知道对于对角线上的元素, 横纵坐标的差值是一样的;
2. 然后我们遍历一遍数组, 把相同对角线上的元素值存放到同一个key的字典中;
3. 然后对字典中每个key对应的value进行排序;
4. 重新遍历一遍数组, 对于每个位置去找到字典中的排序后的value中最小的数值填进去,然后把该最小值弹出;

```py
class Solution(object):
    def diagonalSort(self, mat):
        row = len(mat)
        col = len(mat[0])
        diagonal = defaultdict(list)
        for i in range(row):
            for j in range(col):
                diagonal[i-j].append(mat[i][j])
        for key, val in diagonal.items():
            diagonal[key] = sorted(val, reverse=True)
        for i in range(row):
            for j in range(col):
                mat[i][j] = diagonal[i-j].pop()
        return mat
```

# 2462. Total Cost to Hire K Workers

[力扣题目链接](https://leetcode.cn/problems/total-cost-to-hire-k-workers/description)

使用heap

```py
class Solution(object):
    def totalCost(self, costs, k, candidates):
        res = 0
        q = []
        n = len(costs)
        left = candidates - 1
        right = n - candidates
        # 初始时，我们需要将costs数组中的前candidates和后candidates加入heap
        # 如果两部分candidates重叠了，那么我们一次加入所有元素，避免重复
        if left + 1 < right:
            for i in range(left+1):
                heappush(q, (costs[i],i))
            for i in range(right, n):
                heappush(q, (costs[i],i))
        else:
            for i in range(n):
                heappush(q, (costs[i],i))

        # 选k人，每次选人pop heap元素
        for i in range(k):
            cost, index = heappop(q)
            res += cost
            # 如果candidates还没有重叠，我们可以继续加入candidates
            if left + 1 < right:
                if index < left + 1:
                    left += 1
                    heappush(q, (costs[left],left))
                else:
                    right -= 1
                    heappush(q, (costs[right],right))
            
        return res
```

# 999. Available Captures for Rook

[力扣题目链接](https://lleetcode.cn/problems/available-captures-for-rook/description/)

使用方向数组代替四个for循环
```py
class Solution(object):
    def numRookCaptures(self, board):
        row, col = 0, 0
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == "R":
                    row = i
                    col = j
                    break
        res = 0
        move_x = [0,1,-1,0]
        move_y = [1,0,0,-1]
        for i in range(4):
            step = 0
            while True:
                new_row = row + step * move_x[i]
                new_col = col + step * move_y[i]
                if new_row < 0 or new_row >= len(board) or new_col < 0 or new_col >= len(board[0]) or board[new_row][new_col] == "B":
                    break
                elif board[new_row][new_col] == "p":
                    res += 1
                    break
                step += 1
        return res
```

# 2831. Find the Longest Equal Subarray (有疑问）

[力扣题目链接](https://leetcode.cn/problems/find-the-longest-equal-subarray/description/)

hashmap + sliding window

假定给定区间 [l,r]，此时元素 x 在区间出现的次数为 cnt[x]，则该区间中以要构造以 x 为等值子数组需要删除的元素个数即为 r−l+1−cnt[x]，如果此时 r−l+1−cnt[x]≤k ，则可以构成一个符合题意的等值子数组。

```py
class Solution(object):
    def longestEqualSubarray(self, nums, k):
        count = defaultdict(list)
        for i, num in enumerate(nums):
            count[num].append(i)
        res = 0
        for item in count.values():
            j = 0
            # shrink the window,
            for i in range(len(item)):
                # 不理解这行while
                while item[i]-item[j]-(i-j) > k:
                    j += 1
                res = max(res, i-j+1)
        return res

```

# 1673. Find the Most Competitive Subsequence

[力扣题目链接](https://leetcode.cn/problems/find-the-most-competitive-subsequence/description)

单调栈的使用
```py
class Solution(object):
    def mostCompetitive(self, nums, k):
        res = []
        for i, num in enumerate(nums):
            # 当栈不为空且未来元素数量+当前元素数量大于需要数量且栈尾元素大前元素
            while len(res) > 0 and len(nums)-i+len(res) > k and res[-1] > num:
                res.pop()
            res.append(num)
        return res[:k]
```

# 2903. Find Indices With Index and Value Difference 1

[力扣题目链接](https://leetcode.cn/problems/find-indices-with-index-and-value-difference-i/description)

模拟，i只用模拟到len(nums)-indexDifference就可以，因为要满足indexDifference间隔

```py
class Solution(object):
    def findIndices(self, nums, indexDifference, valueDifference):
        minIndex, maxIndex = 0, 0
        for j in range(indexDifference, len(nums)):
            i = j - indexDifference
            if nums[i] < nums[minIndex]:
                minIndex = i
            elif nums[i] > nums[maxIndex]:
                maxIndex = i 
            if nums[j] - nums[minIndex] >= valueDifference:
                return [j, minIndex]
            elif nums[maxIndex] - nums[j] >= valueDifference:
                return [maxIndex, j]
        return [-1,-1]
```
# 2028. Find Missing Observations

[力扣题目链接](https://leetcode.cn/problems/find-missing-observations/)

```py
class Solution(object):
    def missingRolls(self, rolls, mean, n):
        total = mean * (len(rolls) + n)
        m_sum = sum(rolls)
        # calculate the sum and verfiy it's within the range (possible for dice)
        n_sum = total - m_sum
        if not (n <= n_sum and n_sum <= n*6):
            return []
        res = [0] * n
        # calculate the quotient and remainder, remainder means the extra values needed to achieve sum (remainder is less than quotient)
        quotient = n_sum / n
        remainder = n_sum % n
        for i in range(n):
            if i < remainder:
                res[i] = quotient + 1
            else:
                res[i] = quotient
        return res
        
```

# 2981. Find Longest Special Substring That Occurs Thrice

[力扣题目链接](https://leetcode.cn/problems/find-longest-special-substring-that-occurs-thrice-i/description/)

更新答案时，主要有三种：

最长的 chs[0] 可贡献出 3 个长为 chs[0]−2的子串，并且需要满足 chs[0]>2。 (例如aaaa，有三个aa，长度为2）

当 chs[0] 与 chs[1] 相等时，可贡献出 4 个长为 chs[0]−1 的子串。不等时可由 chs[0] 贡献出 2 个长为 chs[1] 的子串，加上 chs[1] 本身一共 3 个，并且需要满足 chs[0]>1。（例如 aaaa和aaa，可以有三个aaa，长度为3）

可由 chs[0] 与 chs[1] 加上 chs[2] 本身贡献 3 个长为 chs[2] 的子串。（例如aa，a，a，三个a长度为1）


```py
class Solution(object):
    def maximumLength(self, s):
        length = [[]for _ in range(26)]
        count = 0
        # 第一步，用移动窗口记录每单个字符重复的长度
        for i in range(len(s)):
            count += 1
            if i+1 == len(s) or s[i] != s[i+1]:
                ch = ord(s[i]) - ord('a')
                length[ch].append(count)
                count = 0
                # 把长度长的往前推，只需要维护3个就行
                for j in range(len(length[ch])-1,0,-1):
                    if length[ch][j] > length[ch][j-1]:
                        length[ch][j], length[ch][j-1] = length[ch][j-1],length[ch][j]
                    else:
                        break
                if len(length[ch]) > 3:
                    length[ch].pop()
        res = -1
        # 第二步，对于每个字符，更新最长长度（前提是出现三次）
        for i in range(26):
            if len(length[i]) > 0 and length[i][0] > 2:
                res = max(res, length[i][0]-2)
            if len(length[i]) > 1 and length[i][0] > 1:
                res = max(res,min(length[i][0]-1, length[i][1]))
            if len(length[i]) > 2:
                res = max(res, length[i][2])
        return res
```

# 2981. Distribute Candies Among Children

[力扣题目链接](https://leetcode.cn/problems/distribute-candies-among-children-i/description/)

枚举第一个小朋友获得的糖果(最多可以是所有糖果或limit个), 如果n -  i > 2*limit，说明一定会有一个小朋友超出limit，无有效方案。

如果n -  i <= 2*limit, 第二个小朋友至多可以分到min(n - i, limit),同时至少要分到max(0, n - i - limit)来确保第三个小朋友不会超出limit.

那么可取的方案即min(n - i, limit) - max(0, n - i - limit) + 1个
```py
class Solution(object):
    def distributeCandies(self, n, limit):
        res = 0
        for i in range(min(n,limit)+1):
            if n -  i > 2*limit:
                continue
            res += min(n - i, limit) - max(0, n - i - limit) + 1
        return res

```

# 2938. Separate Black and White Balls（优解）

[力扣题目链接](https://leetcode.cn/problems/separate-black-and-white-balls/description/)

不用纠结中间过程，核心就是冒泡排序的思想， 0的前面有多少个1，是一定要做多少次操作的。

```py
class Solution(object):
    def minimumSteps(self, s):
        cnt1=0
        ans=0
        for c in s:
            if c=='1':
                cnt1+=1
            else:
                ans +=cnt1
        return ans
```

# 3040. Maximum Number of Operations With the Same Score II

[力扣题目链接](https://leetcode.cn/problems/maximum-number-of-operations-with-the-same-score-ii/description/)

@cache is a temporary space from which data can be read or written at a very high speed

```py3
class Solution:
    def maxOperations(self, nums: List[int]) -> int:

        @cache
        def dfs(i,j,target):
            if i >= j:
                return 0 
            ans = 0
            if nums[i] + nums[i+1] == target:
                ans = max(ans, 1 + dfs(i+2,j,target))
            if nums[i] + nums[j] == target:
                ans = max(ans, 1 + dfs(i+1, j-1, target))
            if nums[j] + nums[j-1] == target:
                ans = max(ans, 1 + dfs(i, j-2,target))
            return ans
        n = len(nums)
        res = 0
        res = max(res, dfs(0,n-1,nums[0]+nums[1]))
        res = max(res, dfs(0,n-1,nums[0]+nums[-1]))
        res = max(res, dfs(0,n-1,nums[-2]+nums[-1]))
        return res     
```

# 312. Burst Balloons(DP hard)

[力扣题目链接](https://leetcode.cn/problems/burst-balloons/description/)

假设这个区间是个开区间，最左边索引 i，最右边索引 j， “开区间” 的意思是，我们只能戳爆 i 和 j 之间的气球，i 和 j 不要戳。

i 代表 start range，j 代表 end range，k 代表这个区间最后一个被戳破的气球。

那么假设 dp[i][j] 表示开区间 (i,j) 内你能拿到的最多金币。

那么这个情况下你在 (i,j) 开区间得到的金币可以由 dp[i][k] 和 dp[k][j] 进行转移。

由于k是最后一个被戳破的，且i和j不能戳，如果你此刻选择戳爆气球 k，那么你得到的金币数量就是：

total = dp[i][k] + val[i] * val[k] * val[j] + dp[k][j]


```py3
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        # pad 前后处理边界值
        nums = [1] + nums + [1]
        n = len(nums)
        rec = [[0]*(n) for _ in range(n)]
        
        for i in range(n-3,-1,-1):    # i是区间的开始索引，后面至少得有两个元素，所以从n-3开始
            for j in range(i+2,n):    # j是区间的结束索引
                for k in range(i+1,j):    # k为不能戳i，j区间内戳破的气球
                    total = nums[i] * nums[k] * nums[j]
                    total += rec[i][k] + rec[k][j]
                    rec[i][j] = max(rec[i][j],total)
        return rec[0][n-1]
```

# 491. Non-decreasing Subsequences

[力扣题目链接](https://leetcode.cn/problems/non-decreasing-subsequences/description/)

从左往右遍历数组，遍历到一个元素时，如果选择移动到这个位置，那么移动到这个位置所能获得的最大分数，只取决于之前的位置中：

最后移动的元素为偶数时得分的最大值 和 最后移动的元素为奇数时得分的最大值

我们可以用一个长度为 2 的数组 dp 来保存刚才提到的两个值。在求出当前位置可以获得的最大得分后，更新 dp。最后返回所有位置最大得分的最大值即可。


```py3
class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        dp = [-inf, -inf]
        dp[nums[0] % 2] = nums[0]
        for i in range(1,len(nums)):
            parity = nums[i] % 2
            dp[parity] = max(dp[parity]+nums[i], dp[1-parity]-x+nums[i])
        return max(dp[0],dp[1])
```
