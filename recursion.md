# 回溯算法理论基础 

回溯法其实就是暴力查找，并不是什么高效的算法。

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
```
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
            for i in range(index,n - (k-len(path))+2):  #剪枝:如果for循环选择的起始位置之后的元素个数 已经不足 我们需要的元素个数了，那么就没有必要搜索了。
                                                        #比如n=4，k=3，一个空path至少要从2开始搜索，3和4开始的可以剪去。
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

```
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

