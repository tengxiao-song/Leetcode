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

