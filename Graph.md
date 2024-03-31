# DFS
### DFS 框架
```cpp
void dfs(参数) {
    if (终止条件) {
        存放结果;
        return;
    }

    for (选择：本节点所连接的其他节点) {
        处理节点;
        dfs(图，选择的节点); // 递归
        回溯，撤销处理结果
    }
}

```

# 797.所有可能的路径 

[力扣题目链接](https://leetcode.cn/problems/all-paths-from-source-to-target/)

```py
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        ans = []
        path = []
        def dfs(x):
            if x == len(graph) - 1:
                ans.append(path[:])
                return
            for y in graph[x]:
                path.append(y)
                dfs(y)
                path.pop()  #回溯
        path.append(0)
        dfs(0)
        return ans
```

# 695. 岛屿的最大面积 

[力扣题目链接](https://leetcode.cn/problems/max-area-of-island/)

# 1020. 飞地的数量 

[力扣链接](https://leetcode.cn/problems/number-of-enclaves/description/)
