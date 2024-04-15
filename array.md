# 数组理论基础
* **数组下标都是从0开始的。**

* **数组内存空间的地址是连续的**
  
**数组的元素是不能删的，只能覆盖。**
  
**那么二维数组在内存的空间地址是连续的么？**

不同编程语言的内存管理是不一样的，以C++为例，在C++中二维数组是连续分布的。

像Java是没有指针的，同时也不对程序员暴露其元素的地址，寻址操作完全交给虚拟机。二维数组的每一行头结点的地址是没有规则的，更谈不上连续。

# 704. 二分查找

[力扣题目链接](https://leetcode.cn/problems/binary-search/)
## 思路

**这道题目的前提是数组为有序数组**，同时题目还强调**数组中无重复元素**，因为一旦有重复元素，使用二分查找法返回的元素下标可能不是唯一的，这些都是使用二分法的前提条件，当大家看到题目描述满足如上条件的时候，可要想一想是不是可以用二分法了。

二分查找涉及的很多的边界条件，逻辑比较简单，但就是写不好。例如到底是 `while(left < right)` 还是 `while(left <= right)`，到底是`right = middle`呢，还是要`right = middle - 1`呢？

大家写二分法经常写乱，主要是因为**对区间的定义没有想清楚，区间的定义就是不变量**。要在二分查找的过程中，保持不变量，就是在while寻找中每一次边界的处理都要坚持根据区间的定义来操作，这就是**循环不变量**规则。

写二分法，区间的定义一般为两种，左闭右闭即[left, right]，或者左闭右开即[left, right)。

下面我用这两种区间的定义分别讲解两种不同的二分写法。

### 二分法第一种写法

第一种写法，我们定义 target 是在一个在左闭右闭的区间里，**也就是[left, right] （这个很重要非常重要）**。

区间的定义这就决定了二分法的代码应该如何写，**因为定义target在[left, right]区间，所以有如下两点：**

* while (left <= right) 要使用 <= ，因为left == right是有意义的，所以使用 <=
* if (nums[middle] > target) right 要赋值为 middle - 1，因为当前这个nums[middle]一定不是target，那么接下来要查找的左区间结束下标位置就是 middle - 1

```py
class Solution(object):
    def search(self, nums, target):
        left = 0
        right = len(nums)-1
        while left <= right:
            mid = (left+right)/2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return -1
```
### 二分法第二种写法

如果说定义 target 是在一个在左闭右开的区间里，也就是[left, right) ，那么二分法的边界处理方式则截然不同。

有如下两点：

* while (left < right)，这里使用 < ,因为left == right在区间[left, right)是没有意义的
* if (nums[middle] > target) right 更新为 middle，因为当前nums[middle]不等于target，去左区间继续寻找，而寻找区间是左闭右开区间，所以right更新为middle.

# 27. 移除元素

[力扣题目链接](https://leetcode.cn/problems/remove-element/)

## 思路

虽然python可以pop元素，但是用pop的时间不是最优。

**而且要知道数组的元素在内存地址中是连续的，不能单独删除数组中的某个元素，只能覆盖。**
### 双指针法

双指针法（快慢指针法）： **通过一个快指针和慢指针在一个for循环下完成两个for循环的工作。**
* 快指针：指向末尾，删除下标的位置
* 慢指针：指向更新 新数组下标的位置

```py
class Solution(object):
    def removeElement(self, nums, val):
        left = 0
        right = len(nums)-1
        while left <= right:
            if nums[left] == val:
                nums[left], nums[right] = nums[right], nums[left]
                right -= 1
            else:
                left += 1
        return left      
        
```

# 977.有序数组的平方

[力扣题目链接](https://leetcode.cn/problems/squares-of-a-sorted-array/)

## 思路
### 暴力排序
这个时间复杂度是 O(n + nlogn)， 可以说是O(nlogn)的时间复杂度，但为了和下面双指针法算法时间复杂度有鲜明对比，我记为 O(n + nlog n)。
### 双指针法

数组其实是有序的， 只不过负数平方之后可能成为最大数了。

那么数组平方的最大值就在数组的两端，不是最左边就是最右边，不可能是中间。

此时可以考虑双指针法了，i指向起始位置，j指向终止位置。

定义一个新数组result，和A数组一样的大小，让k指向result数组终止位置。

如果`A[i] * A[i] < A[j] * A[j]`  那么`result[k--] = A[j] * A[j];`  。

如果`A[i] * A[i] >= A[j] * A[j]` 那么`result[k--] = A[i] * A[i];` 。

此时的时间复杂度为O(n)，相对于暴力排序的解法O(n + nlog n)还是提升不少的。

# 209.长度最小的子数组

[力扣题目链接](https://leetcode.cn/problems/minimum-size-subarray-sum/)
### 滑动窗口 O(n)

滑动窗口的精妙之处在于根据当前子序列和大小的情况，不断调节子序列的起始位置。从而将O(n^2)的暴力解法降为O(n)。

```py
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        l = len(nums)
        left = 0, right = 0
        min_len = float('inf')
        cur_sum = 0 #当前的累加值
        
        while right < l:
            cur_sum += nums[right]
            while cur_sum >= s: # 当前累加值大于目标值
                min_len = min(min_len, right - left + 1)
                cur_sum -= nums[left]
                left += 1
            right += 1
        return min_len if min_len != float('inf') else 0
```

# 59.螺旋矩阵II

[力扣题目链接](https://leetcode.cn/problems/spiral-matrix-ii/)

**本题并不涉及到什么算法，就是模拟过程，但却十分考察对代码的掌控能力。**


# 1365.有多少小于当前数字的数字

[力扣题目链接](https://leetcode.cn/problems/how-many-numbers-are-smaller-than-the-current-number/)


# 941.有效的山脉数组

[力扣题目链接](https://leetcode.cn/problems/valid-mountain-array/)
## 思路
双指针

# 1207.独一无二的出现次数

[力扣题目链接](https://leetcode.cn/problems/unique-number-of-occurrences/)
## 思路
Hashtable

# 283. 移动零：动态规划：一样的套路，再求一次完全平方数

[力扣题目链接](https://leetcode.cn/problems/move-zeroes/)
## 思路
双指针

# 189. 旋转数组

[力扣题目链接](https://leetcode.cn/problems/rotate-array/)

## 思路
本题是右旋转，步骤如下：

1. 反转整个字符串
2. 反转区间为前k的子串
3. 反转区间为k到末尾的子串

# 724.寻找数组的中心下标

[力扣题目链接](https://leetcode.cn/problems/find-pivot-index/)
## 思路

左半持续增加，右半持续减少并比对

# 34. 在排序数组中查找元素的第一个和最后一个位置

[力扣链接](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

## 思路
**扎扎实实的写两个二分分别找左边界和右边界**
### 寻找右边界
那么这里我采用while (left <= right)的写法，区间定义为[left, right]，即左闭右闭的区间。<br>
用left-1确认右边界是否等于target，并且右边界的index为left-1，因为是由left负责推动边界的。
### 寻找左边界
这里不用确认是否等于target，因为右边界已经保证至少有一个值了。但是左边界的index等于right+1，因为是由right负责推动边界的。

# 922. 按奇偶排序数组II

[力扣题目链接](https://leetcode.cn/problems/sort-array-by-parity-ii/)

# 35.搜索插入位置

[力扣题目链接](https://leetcode.cn/problems/search-insert-position/)
## 思路

先用二分法找target的index。<br>
如果没找到就return最后的left。这里我reture left主要是考虑到insert是把比他大的东西往后推。
并且由于
```
elif nums[mid] < target:
  left = mid + 1
```
left确保了当前位置和之后的都比target大。
