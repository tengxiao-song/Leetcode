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
