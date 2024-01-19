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
