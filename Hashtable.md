# 哈希表理论基础

## 哈希表
**一般哈希表都是用来快速判断一个元素是否出现集合里。**
要枚举的话时间复杂度是O(n)，但如果使用哈希表的话， 只需要O(1)就可以做到。
## 常见的三种哈希结构

* 当我们想使用哈希法来解决问题的时候，我们一般会选择如下三种数据结构。

* 数组
* set （集合）
* map(映射)

这里数组就没啥可说的了，我们来看一下set。

在C++中，set 和 map 分别提供以下三种数据结构，其底层实现以及优劣如下表所示：

| 集合               | 底层实现 | 是否有序 | 数值是否可以重复 | 能否更改数值 | 查询效率 | 增删效率 |
| --- | --- | ---- | --- | --- | --- | --- |
| std::set           | 红黑树   | 有序     | 否               | 否           | O(log n) | O(log n) |
| std::multiset      | 红黑树   | 有序     | 是               | 否           | O(logn)  | O(logn)  |
| std::unordered_set | 哈希表   | 无序     | 否               | 否           | O(1)     | O(1)     |

std::unordered_set底层实现为哈希表，std::set 和std::multiset 的底层实现是红黑树，红黑树是一种平衡二叉搜索树，所以key值是有序的，但key不可以修改，改动key值会导致整棵树的错乱，所以只能删除和增加。

| 映射               | 底层实现 | 是否有序 | 数值是否可以重复 | 能否更改数值 | 查询效率 | 增删效率 |
| --- | --- | --- | --- | --- | --- | --- |
| std::map           | 红黑树   | key有序  | key不可重复      | key不可修改  | O(logn)  | O(logn)  |
| std::multimap      | 红黑树   | key有序  | key可重复        | key不可修改  | O(log n) | O(log n) |
| std::unordered_map | 哈希表   | key无序  | key不可重复      | key不可修改  | O(1)     | O(1)     |


std::unordered_map 底层实现为哈希表，std::map 和std::multimap 的底层实现是红黑树。同理，std::map 和std::multimap 的key也是有序的（这个问题也经常作为面试题，考察对语言容器底层的理解）。

当我们要使用集合来解决哈希问题的时候，优先使用unordered_set，因为它的查询和增删效率是最优的，如果需要集合是有序的，那么就用set，如果要求不仅有序还要有重复数据的话，那么就用multiset。

那么再来看一下map ，在map 是一个key value 的数据结构，map中，对key是有限制，对value没有限制的，因为key的存储方式使用红黑树实现的。

其他语言例如：java里的HashMap ，TreeMap 都是一样的原理。可以灵活贯通。

虽然std::set、std::multiset 的底层实现是红黑树，不是哈希表，std::set、std::multiset 使用红黑树来索引和存储，不过给我们的使用方式，还是哈希法的使用方式，即key和value。所以使用这些数据结构来解决映射问题的方法，我们依然称之为哈希法。 map也是一样的道理。

这里在说一下，一些C++的经典书籍上 例如STL源码剖析，说到了hash_set hash_map，这个与unordered_set，unordered_map又有什么关系呢？

实际上功能都是一样一样的， 但是unordered_set在C++11的时候被引入标准库了，而hash_set并没有，所以建议还是使用unordered_set比较好，这就好比一个是官方认证的，hash_set，hash_map 是C++11标准之前民间高手自发造的轮子。

![哈希表6](https://code-thinking-1253855093.file.myqcloud.com/pics/20210104235134572.png)

## 总结

总结一下，**当我们遇到了要快速判断一个元素是否出现集合里的时候，就要考虑哈希法**。

但是哈希法也是**牺牲了空间换取了时间**，因为我们要使用额外的数组，set或者是map来存放数据，才能实现快速的查找。

如果在做面试题目的时候遇到需要判断一个元素是否出现过的场景也应该第一时间想到哈希法！

# 242.有效的字母异位词

[力扣题目链接](https://leetcode.cn/problems/valid-anagram/)

## 思路
用Counter或者defaultdict, defaultdict不会有key error <br>
Python写法二（没有使用数组作为哈希表，只是介绍defaultdict这样一种解题思路）：

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        from collections import defaultdict
        
        s_dict = defaultdict(int)
        t_dict = defaultdict(int)
        for x in s:
            s_dict[x] += 1
        
        for x in t:
            t_dict[x] += 1
        return s_dict == t_dict
```

# 1002. 查找常用字符 (难点)

[力扣题目链接](https://leetcode.cn/problems/find-common-characters/) 

```python
class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        minfreq = [float("inf")]*26  # 用来统计所有字符串里字符出现的最小频率
        for word in words:
            freq = [0]*26
            for letter in word:
                freq[ord(letter)-ord('a')] += 1
            for i in range(26):
                minfreq[i] = min(minfreq[i],freq[i])  # 更新hash，保证hash里统计26个字符在所有字符串里出现的最小次数
        res = []
        for i in range(26):
            for j in range(minfreq[i]):
                res.append(chr(ord('a')+i))
        return res
```

# 349. 两个数组的交集

[力扣题目链接](https://leetcode.cn/problems/intersection-of-two-arrays/)

# 第202题. 快乐数

[力扣题目链接](https://leetcode.cn/problems/happy-number/)

# 1. 两数之和

[力扣题目链接](https://leetcode.cn/problems/two-sum/)
再来看一下使用数组和set来做哈希法的局限。

* 数组的大小是受限制的，而且如果元素很少，而哈希值太大会造成内存空间的浪费。
* set是一个集合，里面放的元素只能是一个key，而两数之和这道题目，不仅要判断y是否存在而且还要记录y的下标位置，因为要返回x 和 y的下标。所以set 也不能用。

此时就要选择另一种数据结构：map ，map是一种key value的存储结构，可以用key保存数值，用value再保存数值所在的下标。

# 第454题.四数相加II(难点) 

[力扣题目链接](https://leetcode.cn/problems/4sum-ii/)

## 思路
1. 首先定义 一个unordered_map，key放a和b两数之和，value 放a和b两数之和出现的次数。
2. 遍历大A和大B数组，统计两个数组元素之和，和出现的次数，放到map中。
3. 定义int变量count，用来统计 a+b+c+d = 0 出现的次数。
4. 在遍历大C和大D数组，找到如果 0-(c+d) 在map中出现过的话，就用count把map中key对应的value也就是出现次数统计出来。
5. 最后返回统计值 count 就可以了

这里设置了default value to 0，so there is no need to check the keys 
```
rec = defaultdict(lambda : 0)
or
rec = defaultdict(int)
```
# 383. 赎金信

[力扣题目链接](https://leetcode.cn/problems/ransom-note/)

这题除了counter以外，还可以用数组做，counter的map空间消耗比数组大。

# 第15题. 三数之和(难点)

[力扣题目链接](https://leetcode.cn/problems/3sum/)

### 去重逻辑的思考(难点)
```Python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()
        
        for i in range(len(nums)):
            
            # 跳过相同的元素以避免重复
            if i > 0 and nums[i] == nums[i - 1]:
                continue
                
            left = i + 1
            right = len(nums) - 1
            
            while right > left:
                sum_ = nums[i] + nums[left] + nums[right]
                
                if sum_ < 0:
                    left += 1
                elif sum_ > 0:
                    right -= 1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    
                    # 跳过相同的元素以避免重复
                    while right > left and nums[right] == nums[right - 1]:
                        right -= 1
                    while right > left and nums[left] == nums[left + 1]:
                        left += 1
                        
                    right -= 1
                    left += 1
                    
        return result
```

# 第18题. 四数之和

[力扣题目链接](https://leetcode.cn/problems/4sum/)


# 205. 同构字符串

[力扣题目链接](https://leetcode.cn/problems/isomorphic-strings/)
