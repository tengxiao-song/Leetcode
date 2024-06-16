# 回溯算法理论基础 

回溯法其实就是暴力查找，并不是什么高效的算法。

回溯法解决的问题都可以抽象为树形结构，问题可以理解成操作一颗高度有限的树（N叉树）。有终止条件，也有for处理每一个子结点

回溯算法模板框架如下：

```
void backtracking(参数) {
    if (终止条件) {
        存放结果;
        return;
    }

    for (选择：本层集合中元素（树中节点孩子的数量就是集合的大小）) {
        处理节点;
        backtracking(路径，选择列表); // 递归
        回溯，撤销处理结果
    }
}
```

# 第77题. 组合(剪枝)

[力扣题目链接](https://leetcode.cn/problems/combinations/ )

如果for循环选择的起始位置之后的元素个数 已经不足 我们需要的元素个数了，那么就没有必要搜索了。

- if (n+1 - index) < (k - len(path)): return
                                    
比如n=4，k=3，一个空path至少要从2开始搜索，3和4开始的可以剪去。
```py
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        self.res = []
        def dfs(index,n,k,path):
            if len(path) == k:
                self.res.append(path[:])
                return
            for i in range(index,n - (k-len(path))+2):  #剪枝
                path.append(i)
                dfs(i+1,n,k,path)
                path.pop()
        dfs(1,n,k,[])
        return self.res
```

# 216.组合总和III(剪枝）

[力扣题目链接](https://leetcode.cn/problems/combination-sum-iii/)

如果for循环选择的起始位置之后的元素个数 已经不足 我们需要的元素个数了，那么就没有必要搜索了。

比如 k = 3, 至少也要从7开始得到[7,8,9], 从8，9开始没有意义。

```py
class Solution(object):
    def combinationSum3(self, k, n):
        res = []
        def dfs(n,k,path,index):
            if len(path) == k and sum(path) == n:
                res.append(path[:])
            elif len(path) > k or sum(path) > n:
                return
            for i in range(index,11-(k-len(path))):    # k-len(path)为需要的个数
                path.append(i)
                dfs(n,k,path,i+1)
                path.pop()
        dfs(n,k,[],1)
        return res
```

# 17.电话号码的字母组合(又错)

[力扣题目链接](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

单层逻辑：首先要取index指向的数字，并找到对应的字符集（手机键盘的字符集）。然后for循环来处理这个字符集。

这里for循环注可以直接从0开始，而不需要从startIndex开始遍历，因为本题每一个数字代表的是不同集合，也就是求不同集合之间的组合。

```py3
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        phone = {"2":["a","b","c"],"3":["d","e","f"],"4":["g","h","i"],"5":["j","k","l"],"6":["m","n","o"],"7":["p","q","r","s"],"8":["t","u","v"],"9":["w","x","y","z"]}

        res = []
        if not digits: return res

        def dfs(path,index):
            if index == len(digits):
                res.append("".join(path[:]))
                return
            letters = phone[digits[index]]
            for i in range(len(letters)):
                path.append(letters[i])
                dfs(path, index+1)
                path.pop()
        dfs([],0)
        return res
```

# 39. 组合总和

[力扣题目链接](https://leetcode.cn/problems/combination-sum/)

```py3
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort()    #为剪枝作准备

        def dfs(path,index,total):
            if total == target:
                res.append(path[:])
                return
            elif total > target:
                return
            for i in range(index, len(candidates)):
                if total + candidates[i] > target:     #剪枝：后续元素都会更大，直接break
                    break
                path.append(candidates[i])
                total += candidates[i]
                dfs(path, i, total)    # 允许重复选择元素
                total -= candidates[i]
                path.pop()
        dfs([],0,0)
        return res
```

# 40.组合总和II

[力扣题目链接](https://leetcode.cn/problems/combination-sum-ii/)

去重逻辑很关键

```py
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort()    # 为剪枝和去重做准备

        def dfs(path, index, total):
            if total == target:
                res.append(path[:])
                return
            for i in range(index, len(candidates)):
                if i > index and candidates[i] == candidates[i-1]:    # 对同一树层使用过的元素进行跳过
                    continue
                if total + candidates[i] > target:    # 剪枝
                    return
                path.append(candidates[i])
                total += candidates[i]
                dfs(path,i+1, total)
                total -= candidates[i]
                path.pop()
        dfs([], 0, 0)
        return res
```
# 131.分割回文串

[力扣题目链接](https://leetcode.cn/problems/palindrome-partitioning/)

```py
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        def dfs(path, index):
            if index == len(s):
                res.append(path[:])
                return
            for i in range(index, len(s)):
                if s[index:i+1] == s[index:i+1][::-1]:
                    path.append(s[index:i+1])
                    dfs(path, i+1)
                    path.pop()
        dfs([],0)
        return res
```

# 93.复原IP地址

[力扣题目链接](https://leetcode.cn/problems/restore-ip-addresses/)

# 78.子集

[力扣题目链接](https://leetcode.cn/problems/subsets/)

如果把 子集问题、组合问题、分割问题都抽象为一棵树的话，那么组合问题和分割问题都是收集树的叶子节点，而子集问题是找树的所有节点！

# 90.子集II

[力扣题目链接](https://leetcode.cn/problems/subsets-ii/)

就是组合2去重逻辑和子集结合

# 491.递增子序列

[力扣题目链接](https://leetcode.cn/problems/non-decreasing-subsequences/)

```py3
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        res = []
        def dfs(path, index):
            if len(path) >= 2:
                res.append(path[:])
            lset = set()    # 每层定义一个set，来记录使用过的元素，如有重复直接跳过，由于是树层去重且每层有新set不用回溯set
            for i in range(index, len(nums)):
                if (path and nums[i] < path[-1]) or nums[i] in lset:    # 非递增跳过，加树层去重
                    continue
                path.append(nums[i])
                lset.add(nums[i])
                dfs(path, i+1)
                path.pop()
        dfs([], 0)
        return res
```

# 46.全排列

[力扣题目链接](https://leetcode.cn/problems/permutations/)

首先排列是有序的，也就是说 [1,2] 和 [2,1] 是两个集合，这和之前分析的子集以及组合所不同的地方。所以处理排列问题就不用使用startIndex了。

其次一个排列里一个元素只能使用一次.

# 47.全排列 II

[力扣题目链接](https://leetcode.cn/problems/permutations-ii/)

树层去重可以用set判断，也可以排序后跳过数值重复的

```py3
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        def dfs(path, src):
            if len(path) == len(nums):
                res.append(path[:])
            # used = set()
            for i in range(len(src)):
                if i > 0 and src[i] == src[i-1]:
                    continue
                # if src[i] in used:
                    # continue
                path.append(src[i])
                # used.add(src[i])
                dfs(path, src[:i]+src[i+1:])
                path.pop()
        dfs([],nums)
        return res
```

# 332.重新安排行程(困难)

[力扣题目链接](https://leetcode.cn/problems/reconstruct-itinerary/)

```py3
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        res = []
        tickets.sort(key=lambda x:x[1]) # 将所有行程根据destination排序，以便后续查找
        graph = defaultdict(list)    # 将行程转化成graph
        for u, v in tickets:
            graph[u].append(v)

        def dfs(new_graph, s):
            while s in new_graph and len(new_graph[s]) > 0:    # 如该起点还有行程，根据排序取第一个元素(最小lexographic order), 进行dfs
                v = new_graph[s][0]
                new_graph[s].pop(0)
                dfs(new_graph, v)
            res.append(s)    # 加起点加入res，如处理完所有该起点的行程
        dfs(graph, "JFK")
        return res[::-1]    # 由于是处理完所有终点再加入起点，需要反转
```

# 51. N-Queens

[力扣题目链接](https://leetcode.cn/problems/n-queens/description/)

```py3
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        board = [['.'] * n for _ in range(n)]        # 模拟棋盘放皇后

        def dfs(row):
            if row == n:
                res.append([''.join(row) for row in board])        # 需要deepcopy，因为是2D array
                return
            for i in range(n):            # 对当前行遍历所有列，如果有一个位置行列对角都没Q，当前位置可行，进入下一行
                if isValid(row, i):
                    board[row][i] = 'Q'          
                    dfs(row+1)
                    board[row][i] = '.'

        def isValid(row, col):
            # 不用确认横向，因为dfs遍历横向是如果valid会进下一行，不可以会回溯
            for i in range(0,row):        # 确认上方列
                if board[i][col] == 'Q':
                    return False
            i = row-1
            j = col-1
            while i >=0 and j >= 0:        # 确认左上方对角线
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            i = row-1
            j = col + 1
            while i >= 0 and j < n:        # 确认右上方对角线
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            return True
        dfs(0)
        return res
```
