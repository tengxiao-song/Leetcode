# 705 Design Hashset

[力扣题目链接](https://leetcode.cn/problems/design-hashset/description/)

### 方法1: array

创建一个巨大的array，然后每次add，remove和contains都对key对应的下标进行处理。时间O(1), 空间浪费。

### 方法2: close hashing

时间复杂度：O(n/b), 其中 n 为哈希表中的元素数量，b 为链表的数量。空间复杂度：O(n+b)

```py
class MyHashSet(object):

    def __init__(self):
        self.base = 769
        self.nums = [[]]*self.base

    def add(self, key):
        """
        :type key: int
        :rtype: None
        """
        new_key = key % self.base
        val = self.nums[new_key]
        if key not in val:
            val.insert(0,key)


    def remove(self, key):
        """
        :type key: int
        :rtype: None
        """
        new_key = key % self.base
        val = self.nums[new_key]
        if key in val:
            val.remove(key)


    def contains(self, key):
        """
        :type key: int
        :rtype: bool
        """
        new_key = key % self.base
        val = self.nums[new_key]
        return key in val
```

