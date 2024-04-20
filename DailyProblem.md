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

[力扣题目链接]([https://leetcode.cn/problems/add-one-row-to-tree/description/](https://leetcode.cn/problems/smallest-string-starting-from-leaf/description/))

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
