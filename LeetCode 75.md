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

# 2542. Maximum Subsequence Score

[力扣题目链接](https://leetcode.cn/problems/maximum-subsequence-score/description/)

子序列无需连续

```py3
class Solution:
    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        combine = sorted(zip(nums1,nums2), key=lambda x:-x[1]) # 合并数组一起排序，按nums2从大到小
        h = combine[:k]    # 截取前k个
        heap = [i for i, _ in h] #固定为 k 的最小堆来做，如果当前元素大于堆顶，就替换堆顶，这样可以让堆中元素之和变大
        heapq.heapify(heap)
        total = sum(heap)
        least = h[k-1][1]    # 当前范围内最小的元素
        res = total * least
        for x, y in combine[k:]:
            if x > heap[0]:    # 如果当前元素大于heap最小元素，代表可以尝试选择该下标，从新计算最大sum和least
                total = total + x - heap[0]
                least = min(least,y)
                res = max(res, total * least)
                heapq.heapreplace(heap,x)
        return res
```

# 2462. Total Cost to Hire K Workers

[力扣题目链接](https://leetcode.cn/problems/total-cost-to-hire-k-workers/description/)

```py3
class Solution(object):
    def totalCost(self, costs, k, candidates):
        res = 0
        q = []
        n = len(costs)
        left = candidates - 1
        right = n - candidates
        if left + 1 < right:
            for i in range(left+1):        # 加入值和index，用来判断是哪边加入的
                heappush(q, (costs[i],i))
            for i in range(right, n):
                heappush(q, (costs[i],i))
        else:
            for i in range(n):
                heappush(q, (costs[i],i))
        for i in range(k):
            cost, index = heappop(q)
            res += cost
            if left + 1 < right:    # left + 1 < right 避免出现重复元素
                if index < left + 1:
                    left += 1
                    heappush(q, (costs[left],left))
                else:
                    right -= 1
                    heappush(q, (costs[right],right))
        return res
```

# 875. Koko Eating Bananas

[力扣题目链接](https://leetcode.cn/problems/koko-eating-bananas/description/)

这里不二分index，而是二分时间。

while (left <= right) 要使用 <= ，因为left == right是有意义的，所以使用 <=
if (nums[middle] > target) right 要赋值为 middle - 1，因为当前这个nums[middle]一定不是target，那么接下来要查找的左区间结束下标位置就是 middle - 1

```py3
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        n = len(piles)
        low = 1
        high = max(piles)
        while low <= high:
            speed = (low + high) // 2
            hours = 0
            for i in range(n):
                hours += math.ceil(piles[i] / speed)
            if hours <= h:
                high = speed - 1
            else:
                low = speed + 1
        return low
```

# 136. Single Number

[力扣题目链接](https://leetcode.cn/problems/single-number/description/)

异或XOR：两个相同的数异或结果为0（0^0=0, 1^1=0),两个不同数异或为它本身(1^0=1,0^1=1)

异或运算满足交换律和结合律

```py3
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res = res ^ num
        return res
```

# 1268. Search Suggestions System

[力扣题目链接](https://leetcode.cn/problems/search-suggestions-system/description/)

```py3
class Trie:
    def __init__(self):
        self.child = dict()    # 下一层入口
        self.words = []    # 每一层最多保留3个词
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        root = Trie()
        for product in products:
            temp = root
            for ch in product:
                if ch not in temp.child:
                    temp.child[ch] = Trie()
                temp = temp.child[ch]
                temp.words.append(product)
                temp.words.sort()
                if len(temp.words) > 3:
                    temp.words.pop()
        res = []
        flag = False # 表示没有匹配的前缀，可以直接返回空数组
        for i in searchWord:
            if flag or i not in root.child:
                flag = True
                res.append([])
            else:
                root = root.child[i]
                res.append(root.words)
        return res
```

# 128. Longest Consecutive Sequence

用hashset达成O(n)复杂度，**简单来说就是每个数都判断一次这个数是不是连续序列的开头那个数**，如果是从下往上进行计数。

```py3
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        res = 0
        collect = set(nums)
        for i in collect:
            if i - 1 in collect:
                continue
            else:
                temp = 1
                while i + 1 in collect:
                    temp += 1
                    i += 1
                res = max(res,temp)
        return res
```

# 560. Subarray Sum Equals K


简单来说，就是随着i递增，pre作为当前到i的数组和，查看map里面有没有值为pre-k的，有的话刚好pre -（pre-k）就等于k，满足需求，而map里的value值代表这个点的个数有几个，即，不同的数组段（子序列）有几个，然后加入count，最后把到下标为i的数组和存进map里，然后进入下一个i。


```py3
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        res = 0
        count = defaultdict(int)
        count[0] = 1 # basecase
        pre = 0
        for i in range(len(nums)):
            pre += nums[i]
            if pre - k in count:
                res += count[pre-k]
            count[pre] += 1
        return res
```

# 76. Minimum Window Substring

```py3
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        res_left, res_right = -1, len(s)
        left = 0
        count_s = defaultdict(int)
        count_t = Counter(t)
        less = len(count_t)      # 记录有less个键值对, 如less为0代表当前count_s覆盖了count_t
        for right, val in enumerate(s):
            count_s[val] += 1        # 移动右节点，把每个元素加入s
            if count_s[val] == count_t[val]:    # 标记一个键值对满足
                less -= 1
            while less == 0:        # count_s >= count_t(如都满足，移动左节点)
                if right - left < res_right - res_left:    # 尝试更新更小的区间
                    res_left, res_right = left, right
                if count_s[s[left]] == count_t[s[left]]:    # 如左节点是t需要的字符，增加less
                    less += 1
                count_s[s[left]] -= 1        # 缩小左节点
                left += 1
        return "" if res_left == -1 else s[res_left:res_right+1]
```

# 189. Rotate Array

```py3
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k = k % n 
        def reverse(left,right):
            while left < right:
                nums[left],nums[right] = nums[right],nums[left]    # 手动翻转，使用切片会使用内存
                left += 1
                right -= 1
        reverse(0,n-1)    # 翻转整个数组
        reverse(0,k-1)    # 翻转前k个
        reverse(k,n-1)    # 翻转后n-k个
```

# 41. First Missing Positive

把合法范围内的数字换到对应的下标上，比如数字1要在下标0上，数字2要在下标1上。这样遍历所有下标就知道缺少哪个数了

```py3
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i]-1] != nums[i]:    # 新的nums[i]可能还不在正确的位子上，所以是while
                nums[nums[i]-1],nums[i] = nums[i],nums[nums[i]-1]
        for i in range(n):
            if nums[i] != i+1:
                return i+1
        return n + 1
```

# 240. Search a 2D Matrix II

从右上角看是一棵二叉搜索树

```py3
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        n, m = len(matrix), len(matrix[0])
        row, col = 0, len(matrix[0])-1
        while row >= 0 and row < n and col >= 0 and col <= m:
            if target < matrix[row][col]:
                col -= 1
            elif target > matrix[row][col]:
                row += 1
            else:
                return True
        return False
```
