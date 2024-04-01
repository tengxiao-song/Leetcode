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

# 130. 被围绕的区域 

[题目链接](https://leetcode.cn/problems/surrounded-regions/)

# 417. 太平洋大西洋水流问题 

[题目链接](https://leetcode.cn/problems/pacific-atlantic-water-flow/)

那么我们可以从太平洋边上的节点 逆流而上，将遍历过的节点都标记上。 从大西洋的边上节点 逆流而长，将遍历过的节点也标记上。 然后两方都标记过的节点就是既可以流太平洋也可以流大西洋的节点。

从太平洋边上节点出发，如图：  

![图一](https://code-thinking-1253855093.file.myqcloud.com/pics/20220722103029.png)

从大西洋边上节点出发，如图：  

![图二](https://code-thinking-1253855093.file.myqcloud.com/pics/20220722103330.png)

```py
class Solution(object):
    def pacificAtlantic(self, heights):
        m, n = len(heights), len(heights[0])

        def search(starts):
            visited = set()
            def dfs(x,y):
                if (x, y) in visited:
                    return
                visited.add((x, y))
                for nx, ny in ((x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)):
                    if 0 <= nx < m and 0 <= ny < n and heights[nx][ny] >= heights[x][y]:    #只考虑高度相同或更大的单元格
                        dfs(nx, ny)
            for x, y in starts:
                dfs(x, y)
            return visited

        pacific = [(0, i) for i in range(n)] + [(i, 0) for i in range(1, m)] # 第一行第一列坐标
        atlantic = [(m - 1, i) for i in range(n)] + [(i, n - 1) for i in range(m - 1)]  # 最后一行最后一列坐标
        return list(map(list, search(pacific) & search(atlantic))) # set intersection, 然后把每个tuple map成list
```

