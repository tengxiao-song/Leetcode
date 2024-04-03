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

# 827.最大人工岛 

[力扣链接](https://leetcode.cn/problems/making-a-large-island/)

第一步：一次遍历地图，得出各个岛屿的面积，并做编号记录。可以使用map记录，key为岛屿编号，value为岛屿面积.

![](https://code-thinking-1253855093.file.myqcloud.com/pics/20220829105644.png)

第二步：再遍历地图，遍历0的方格（因为要将0变成1），并统计该1（由0变成的1）周边岛屿面积，将其相邻面积相加在一起，遍历所有 0 之后，就可以得出 选一个0变成1 之后的最大面积。

![](https://code-thinking-1253855093.file.myqcloud.com/pics/20220829105249.png)

```py
class Solution(object):
    def largestIsland(self, grid):
        visited = set() #标记访问过的位置
        n = len(grid)
        res = 0
        self.island_size = 0 #用于保存当前岛屿的尺寸
        directions = [[0,1], [0, -1], [1,0],[-1,0]] #四个方向
        islands_size = defaultdict(int) #保存每个岛屿的尺寸

        def dfs(island_num, r, c):
            visited.add((r,c))
            grid[r][c] = island_num  #访问过的位置标记为岛屿编号
            self.island_size += 1
            for i in range(4):
                nexrR = r + directions[i][0]
                nextC = c + directions[i][1]
                if (nexrR not in range(n) or nextC not in range(n) or (nexrR, nextC) in visited):
                    continue    #越界或以访问坐标continue
                if grid[nexrR][nextC] == 1:  #遇到有效坐标，进入下一个层搜索
                    dfs(island_num, nexrR, nextC)
        island_num = 2    #初始岛屿编号设为2， 因为grid里的数据有0和1， 所以从2开始编号
        all_land = True
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    all_land = False    #地图里不全是陆地
                if (i, j) not in visited and grid[i][j] == 1:
                    self.island_size = 0    #遍历每个位置前重置岛屿尺寸为0
                    dfs(island_num, i , j)
                    islands_size[island_num] = self.island_size    #保存当前岛屿尺寸
                    island_num += 1    #下一个岛屿编号加一
        if all_land: return n*n    #如果全是陆地， 返回地图面积
        count = 0
        visited_island = set()    #保存访问过的岛屿编号
        for i in range(n):
            for j in range(n):
                if grid[i][j] == 0:
                    count = 1     #把由0转换为1的位置计算到面积里
                    visited_island.clear()
                    for x in range(4):
                        nearR = i + directions[x][0]
                        nearC = j + directions[x][1]
                        if nearR not in range(n) or nearC not in range(n): #周围位置越界
                            continue
                        if grid[nearR][nearC] in visited_island:  #岛屿已访问
                            continue
                        count += islands_size[grid[nearR][nearC]]    #累加连在一起的岛屿面积
                        visited_island.add(grid[nearR][nearC])    #标记当前岛屿已访问
                    res = max(res, count)
        return res
```

# 127. 单词接龙

[力扣题目链接](https://leetcode.cn/problems/word-ladder/)

* 给你两个单词 beginWord 和 endWord 和一个字典 wordList ，找到从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0。

以示例1为例，从这个图中可以看出 hit 到 cog的路线，不止一条，有三条，一条是最短的长度为5，两条长度为6。

![](https://code-thinking-1253855093.file.myqcloud.com/pics/20210827175432.png)

首先题目中并没有给出点与点之间的连线，而是要我们自己去连，条件是字符只能差一个，所以判断点与点之间的关系，要自己判断是不是差一个字符，如果差一个字符，那就是有链接。

然后就是求起点和终点的最短路径长度，**这里无向图求最短路，广搜最为合适，广搜只要搜到了终点，那么一定是最短的路径**。因为广搜就是以起点中心向四周扩散的搜索。

```py
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        word_set = set(wordList)    #set结构，查找更快一些
        if endWord not in word_set: return 0    #endWord没有，不存在path直接返回0
        mapping = {beginWord:1}    #记录word是否访问过
        queue = deque([beginWord])
        while queue:
            word = queue.popleft()
            path = mapping[word]    #这个word的路径长度
            for i in range(len(word)):
                word_list = list(word)    #把word分成char准备替换
                for j in range(26):
                    word_list[i] = chr(ord('a')+j)    #每个char从a到z进行替换
                    new_word = "".join(word_list)
                    if new_word == endWord:    #如果新词是endword，返回当前长度+1
                        return path + 1
                    if new_word in word_set and new_word not in mapping:    #如果新词在set里且没访问过，记录新词路径长度，入queue（等同于访问一个vertex）
                        mapping[new_word] = path + 1
                        queue.append(new_word)
        return 0
```
