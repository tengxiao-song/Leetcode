# 344.反转字符串

[力扣题目链接](https://leetcode.cn/problems/reverse-string/)

# 541. 反转字符串II

[力扣题目链接](https://leetcode.cn/problems/reverse-string-ii/)

若该子串长度不足k，则反转整个子串。
```
  res = list(s)
  for i in range(0,len(s),2*k):
    res[i:i+k] = reversed(res[i:i+k])
  return "".join(res)
```
# 151.翻转字符串里的单词

[力扣题目链接](https://leetcode.cn/problems/reverse-words-in-a-string/)
**不要使用辅助空间，空间复杂度要求为O(1)。** python必须使用空间，不能直接做string assignment。

解题思路如下：

* 移除多余空格
* 将整个字符串反转
* 将每个单词反转

```py
class Solution(object):
    def reverseWords(self, s):
        left = 0
        right = len(s) - 1
        # delete leading white space
        while left <= right and s[left] == " ":
            left += 1
        # delete trailing white space
        while right >= left and s[right] == " ":
            right -= 1
        words = []
        # delete extra middle white space
        while left <= right:
            if s[left] != " " or words[-1] != " ":
                words.append(s[left])
            left += 1
        # reverse the entire string
        words = words[::-1]
        words.append(" ")
        left = 0
        # reverse each word
        for i in range(len(words)):
            if words[i] == " ":
                words[left:i] = words[left : i][::-1]
                left = i + 1
        return "".join(words)[:-1]

```
# 28. Find the Index of the First Occurrence in a String (KMP)!!!

[力扣题目链接](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/)

给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。

如果loop haystack用每个字符作为头开始匹配needle的话，时间复杂度是O(n * m).

使用KMP算法可以有效减少时间复杂度。

KMP的经典思想就是:**当出现字符串不匹配时，可以记录一部分之前已经匹配的文本内容，利用这些信息避免从头再去做匹配。**

定义： 前缀是指不包含最后一个字符的所有以第一个字符开头的连续子串。后缀是指不包含第一个字符的所有以最后一个字符结尾的连续子串。

我们要找到最大相同前后缀的长度，并存入前缀表里，这样如果有没能匹配上的，我们就可以按照前缀表的值去避免从头做匹配。

> 例子：needle = aabaaf， haystack = aabaabaaf \
> 我们直接找needle的最大相同前后缀长度，并存入前缀表 \
> a 没有前缀，没有后缀，最大相同前后缀长度为0 \
> aa 前缀: a, 后缀：a，最大相同前后缀长度为1 \
> aab 前缀：a，aa，后缀：b，ab，最大相同前后缀长度为0 \
> aaba 前缀：a，aa，aab，后缀：a，ba，aba，最大相同前后缀长度为 1 \
> aabaa，前缀：a，aa，aab，aaba，后缀：a，aa，baa，abaa，最大相同前后缀长度为2 \
> aabaaf 前缀：a，aa，aab，aaba，aabaa，后缀：f，af，aaf，baaf，abaaf，最大相同前后缀长度为0 \
> 这里needle[5]和haystack[5]不匹配，我们可以直接获取前一位的当前最大相同前后缀长度，也就是2，并从index2 needle[2]=b开始匹配

```py
class Solution(object):
    def strStr(self, haystack, needle):
        # 搭建前缀表
        next = [0] * len(needle)
        # j is the end of the suffix, i is end of prefix
        j = 0
        for i in range(1, len(needle)):
            while j > 0 and needle[i] != needle[j]:
                j = next[j-1]  # 如果前后缀不匹配，回退j直到字符匹配或到0为止
            if needle[i] == needle[j]:
                j += 1        # 如果前后缀匹配，更新j和前缀表
                next[i] = j
        # 比较两个字符串
        j = 0
        for i in range(len(haystack)):
            while j > 0 and haystack[i] != needle[j]:
                j = next[j-1]
            if haystack[i] == needle[j]:
                j += 1
            if j == len(needle):
                return i - len(needle) + 1
        return -1

```

# 459.重复的子字符串 (KMP不懂）

[力扣题目链接](https://leetcode.cn/problems/repeated-substring-pattern/)
KMP


# 925.长按键入
[力扣题目链接](https://leetcode.cn/problems/long-pressed-name/)


# 844.比较含退格的字符串

[力扣题目链接](https://leetcode.cn/problems/backspace-string-compare/)
