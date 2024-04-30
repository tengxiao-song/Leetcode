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

