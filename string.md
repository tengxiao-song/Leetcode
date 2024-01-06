# 344.反转字符串

[力扣题目链接](https://leetcode.cn/problems/reverse-string/)

# 541. 反转字符串II(难点)

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
**不要使用辅助空间，空间复杂度要求为O(1)。**
所以解题思路如下：

* 移除多余空格
* 将整个字符串反转
* 将每个单词反转
# 28. 实现 strStr() (KMP不懂)

[力扣题目链接](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/)
KMP的经典思想就是:**当出现字符串不匹配时，可以记录一部分之前已经匹配的文本内容，利用这些信息避免从头再去做匹配。**
# 459.重复的子字符串 (KMP不懂）

[力扣题目链接](https://leetcode.cn/problems/repeated-substring-pattern/)
KMP


# 925.长按键入
[力扣题目链接](https://leetcode.cn/problems/long-pressed-name/)


# 844.比较含退格的字符串

[力扣题目链接](https://leetcode.cn/problems/backspace-string-compare/)
