# Array/String

## 1071. Greatest Common Divisor of Strings

[力扣题目链接](https://leetcode.cn/problems/greatest-common-divisor-of-strings/description/)

需要知道一个性质：如果 str1 和 str2 拼接后等于 str2和 str1 拼接起来的字符串（注意拼接顺序不同），那么一定存在符合条件的字符串 X。

```py
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        if str1 + str2 == str2 + str1:
            gcd = math.gcd(n,m)
            return str2[:gcd]
        return ""
```

## 605. Can Place Flowers

[力扣题目链接](https://leetcode.cn/problems/can-place-flowers/description/)

数组前后+0，处理边界

```py3
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        target = n
        n = len(flowerbed)
        flowerbed = [0] + flowerbed + [0]
        for i in range(1,n+1):
            if flowerbed[i-1] == 0 and flowerbed[i] == 0 and flowerbed[i+1] == 0:
                target -= 1
                flowerbed[i] = 1
        return target <= 0
```

## 238. Product of Array Except Self

[力扣题目链接](https://leetcode.cn/problems/product-of-array-except-self/description/)

利用索引左侧所有数字的乘积和右侧所有数字的乘积（即前缀与后缀）相乘得到答案。

```py3
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res1 = [1] * n
        res2 = [1] * n
        for i in range(1, n):
            res1[i] = res1[i-1] * nums[i-1]
        for i in range(n-2,-1,-1):
            res2[i] = res2[i+1] * nums[i+1]
        res = [1] * n
        for i in range(n):
            res[i] = res1[i] * res2[i]
        return res
```

## 334. Increasing Triplet Subsequence

[力扣题目链接](https://leetcode.cn/problems/increasing-triplet-subsequence/description/)

为了找到递增的三元子序列，first 和 second 应该尽可能地小，此时找到递增的三元子序列的可能性更大.

如果遍历过程中遇到的所有元素都大于 first，则当遇到 num > second 时，first 一定出现在 second 的前面，second 一定出现在 num 的前面，(first,second,num) 即为递增的三元子序列。

如果遍历过程中遇到小于 first 的元素，则会用该元素更新 first，虽然更新后的 first 出现在 second 的后面，但是在 second 的前面一定存在一个元素 first’ 小于 second，因此当遇到 num>second 时，(first’,second,num) 即为递增的三元子序列。


```py3
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        n = len(nums)
        first = nums[0]
        second = float("inf")
        for num in nums:
            if num > second:
                return True
            if num > first:
                second = num
            else:
                first = num
        return False
```

## 394. Decode String

[力扣题目链接](https://leetcode.cn/problems/decode-string/description/)

```py3
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        for i in s:
            if i != "]":        # 之前写的isalnum判断，导致遇到左括号也进行处理的bug
                stack.append(i)
            else:
                word = ""
                while stack[-1].isalpha():
                    word = stack.pop() + word
                stack.pop()    # 删除左括号
                num = 0 
                t = 1
                while stack and stack[-1].isnumeric():
                    num += int(stack.pop()) * t
                    t *= 10
                stack.append(word * num)
        return "".join(stack)
```

## 649. Dota2 Senate

[力扣题目链接](https://leetcode.cn/problems/dota2-senate/description/)

```py3
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        senate = list(senate)
        n = len(senate)
        R, D = True, True
        flag = 0 # 大于0表示可以跳过r，小于0表示可以跳过d
        while R and D:
            R = False
            D = False
            for i in range(n):
                if senate[i] == "R":
                    if flag > 0:
                        senate[i] = '0'
                    else:
                        R = True
                    flag -= 1
                elif senate[i] == "D":
                    if flag < 0:
                        senate[i] = '0'
                    else:
                        D = True
                    flag += 1
        return "Radiant" if R else "Dire"
```

# 236. 二叉树的最近公共祖先 (又错)

[力扣题目链接](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

后续遍历

```py
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        # 1. 先看根节点是不是祖先
        if not root or root == p or root == q:
            return root
        # 如果根节点是祖先，有没有更近的祖先呢
        left = self.lowestCommonAncestor(root.left,p,q)
        right = self.lowestCommonAncestor(root.right,p,q)
        # 如果有的话显然只会在一侧 判断一下
        if not left: return right    #case:都在右
        if not right: return left   #case:都在左
        # 如果没有更近的，默认还是返回root
        return root
```

# 399. Evaluate Division (hard)

[力扣题目链接](https://leetcode.cn/problems/evaluate-division/description/)

```py3
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        # 生成存储变量所构成的图结构
        graph = {}  
        for (s, e), v in zip(equations, values):
            if s not in graph:
                graph[s] = {}   # 存储邻节点的哈希表
            graph[s][e] = v     # 生成一条s指向e，权重为v的路径，表示 s / e = v
            if e not in graph:
                graph[e] = {}
            graph[e][s] = 1 / v # 生成一条反向路径，权重为1 / v，表示 e / s = 1 /v
            graph[s][s] = 1.0   # 生成一个指向自己、权重为1的路径，表示自己除自己等于1
            graph[e][e] = 1.0
        
        n = len(queries)    
        ans = [-1.0] * n    # 答案列表，初始都为-1表示未定义

        # 对于每个query，寻找从起点qx到终点qy的最短路径，并计算权重积
        for i, (qx, qy) in enumerate(queries):
            if qx not in graph or qy not in graph: continue  # 未出现的变量，跳过处理
            queue = [[qx, 1.0]]     # 初始将起点节点入队
            visited = set([qx])     # 存储已处理的节点；将qx放入列表表示存储整个字符串，否则会将字符串看成一个序列存储每个字母
            while queue:
                node, mul = queue.pop(0)    # 获取当前处理的节点node以及到该节点所得到的权重积mul
                for neighbor, weight in graph[node].items():
                    # 枚举该节点的所有邻节点
                    if neighbor == qy:
                        ans[i] = mul * weight   # 找到终点，更新权重积后存储到答案并退出查找
                        break
                    if neighbor not in visited: # 找到一个未处理的邻节点加入队列
                        visited.add(neighbor)
                        queue.append([neighbor, mul * weight])  # 将未处理的邻节点及到达该节点时的权重积加入队列
        return ans                
```

# 1926. Nearest Exit from Entrance in Maze

[力扣题目链接](https://leetcode.cn/problems/nearest-exit-from-entrance-in-maze/description/)

```py3
class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        q = deque([[entrance[0],entrance[1],0]])
        location = [[1,0],[-1,0],[0,-1],[0,1]]
        visited = set()
        visited.add((entrance[0],entrance[1]))
        while q:
            item = q.popleft()
            x,y,step = item[0],item[1],item[2]
            maze[x][y] = "+"
            if (x == 0 or x == len(maze)-1 or y == 0 or y == len(maze[0])-1) and not (x==entrance[0] and y == entrance[1]):
                return step
            for i in range(4):
                newX = x + location[i][0]
                newY = y + location[i][1]
                if newX >= 0 and newX <= len(maze)-1 and newY >= 0 and newY <= len(maze[0])-1 and maze[newX][newY] == "." and (newX,newY) not in visited:
                    q.append([newX,newY,step+1])
                    visited.add((newX,newY))
        return -1
```

