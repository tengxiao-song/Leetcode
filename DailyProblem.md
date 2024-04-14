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

# 706 Design HashTable

[力扣题目链接](https://leetcode.cn/problems/design-hashmap/)

「设计哈希映射」与「设计哈希集合」解法接近，唯一的区别在于我们存储的不是 key 本身，而是 (key,value)。

```py
class MyHashMap(object):

    def __init__(self):
        self.magic_num = 1009
        self.res = [[]]*self.magic_num

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        index = key % self.magic_num
        for i, (k,v) in enumerate(self.res[index]):    #use enumerate to modify, or else only able to modify copies of k,v
            if k == key:
                self.res[index][i] = (key,value)
                return
        self.res[index].append((key,value))

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        index = key % self.magic_num
        for k, v in self.res[index]:
            if k == key:
                return v
        return -1

    def remove(self, key):
        """
        :type key: int
        :rtype: None
        """
        index = key % self.magic_num
        for i, (k, v) in enumerate(self.res[index]):
            if k == key:
                self.res[index].pop(i)
                return
```

