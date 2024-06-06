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

# 216.组合总和III

[力扣题目链接](https://leetcode.cn/problems/combination-sum-iii/)

# 17.电话号码的字母组合(又错)

[力扣题目链接](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

index很关键

```py
class Solution(object):
    def letterCombinations(self, digits):
        self.book = {'2':["a","b",'c'],'3':['d','e','f'],'4':['g','h','i'],'5':['j','k','l'],'6':['m','n','o'],'7':['p','q','r','s'],'8':['t','u','v'],'9':['w','x','y','z']}
        self.res = []
        if not digits:return []
        def dfs(digits,index,path):
            if index == len(digits):
                self.res.append(path)
                return
            letters = self.book[digits[index]]
            for letter in letters:
                dfs(digits,index+1,path+letter)
        dfs(digits,0,"")
        return self.res
```

# 39. 组合总和

[力扣题目链接](https://leetcode.cn/problems/combination-sum/)

# 40.组合总和II(又错)

[力扣题目链接](https://leetcode.cn/problems/combination-sum-ii/)

去重逻辑很关键

```py
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        self.res = []
        def dfs(candidates,target,index,path):
            if sum(path) == target:
                self.res.append(path[:])
            elif sum(path) > target:
                return
            else:
                for i in range(index,len(candidates)):
                    //remove duplicates!!!
                    if i > index and candidates[i] == candidates[i-1]:
                        continue
                    dfs(candidates,target,i+1,path+[candidates[i]])
        dfs(candidates,target,0,[])
        return self.res
```
# 131.分割回文串

[力扣题目链接](https://leetcode.cn/problems/palindrome-partitioning/)

```py
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        self.res = []
        def dfs(s,path):
            if not s:
                self.res.append(path[:])
            for i in range(1,len(s)+1):
                if s[:i] == s[:i][::-1]:
                    dfs(s[i:],path+[s[:i]])
        dfs(s,[])
        return self.res
```

# 93.复原IP地址

[力扣题目链接](https://leetcode.cn/problems/restore-ip-addresses/)

# 78.子集

[力扣题目链接](https://leetcode.cn/problems/subsets/)

# 90.子集II

[力扣题目链接](https://leetcode.cn/problems/subsets-ii/)

# 491.递增子序列

[力扣题目链接](https://leetcode.cn/problems/non-decreasing-subsequences/)

```py
class Solution(object):
    def findSubsequences(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        self.res = []
        def dfs(nums,path,index):
            if not nums:
                return
            if len(path) >= 2:
                if path[-1] >= path[-2]:
                    self.res.append(path[:])
                else:
                    return        #if not in ascending order, return immedaitely and aborts the later recursion
            for i in range(index,len(nums)):
                if nums[i] in nums[index:i]:        #prevents duplicates, because we can't order the nums
                    continue
                dfs(nums,path+[nums[i]],i+1)
        dfs(nums,[],0)
        return self.res
```

# 46.全排列

[力扣题目链接](https://leetcode.cn/problems/permutations/)

# 47.全排列 II

[力扣题目链接](https://leetcode.cn/problems/permutations-ii/)

# 332.重新安排行程(困难)

[力扣题目链接](https://leetcode.cn/problems/reconstruct-itinerary/)

```py
class Solution:
    def findItinerary(self, tickets):
        targets = collections.defaultdict(list)
        for ticket in tickets:
            targets[ticket[0]].append(ticket[1])    #把出发地和可以去到的终点记录在map里
        for val in targets.values():
            val.sort()    # 对目的地列表进行排序, 保证lexical order
        path = ['JFK']
        ticketsLen = len(tickets) + 1
        
        def dfs(targets,path,ticketsLen):
            if len(path) == ticketsLen:    
                return True            #找到有效行程
            airport = path[-1]    #更新出发点
            destination = targets[airport]        #更新所有终点
            for i, dest in enumerate(destination):    
                targets[airport].pop(i)    # 标记已使用的机票，避免重复使用
                path.append(dest)        # 添加目的地到路径
                if dfs(targets,path,ticketsLen):
                    return True        #找到有效行程，直接返回，因为sorted了
                targets[airport].insert(i,dest)    #没找到就恢复机票
                path.pop()        #移除目的地
            return False        # 没有找到有效行程
        dfs(targets,path,ticketsLen)
        return path
```

