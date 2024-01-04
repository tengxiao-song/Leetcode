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
