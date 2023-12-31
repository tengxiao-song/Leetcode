# 关于链表，你该了解这些！

什么是链表，链表是一种通过指针串联在一起的线性结构，每一个节点由两部分组成，一个是数据域一个是指针域（存放指向下一个节点的指针），最后一个节点的指针域指向null（空指针的意思）。
## 链表的存储方式
链表是通过指针域的指针链接在内存中各个节点。

所以链表中的节点在内存中不是连续分布的 ，而是散乱分布在内存中的某地址上，分配机制取决于操作系统的内存管理。
## 链表的定义
```cpp
// 单链表
struct ListNode {
    int val;  // 节点上存储的元素
    ListNode *next;  // 指向下一个节点的指针
    ListNode(int x) : val(x), next(NULL) {}  // 节点的构造函数
};
```
## 性能分析

![链表-链表与数据性能对比](https://code-thinking-1253855093.file.myqcloud.com/pics/20200806195200276.png)

# 203.移除链表元素

[力扣题目链接](https://leetcode.cn/problems/remove-linked-list-elements/)

## 思路
* **直接使用原来的链表来进行删除操作。**
* **设置一个虚拟头结点在进行删除操作。**

# 707.设计链表

[力扣题目链接](https://leetcode.cn/problems/design-linked-list/)

# 206.反转链表

[力扣题目链接](https://leetcode.cn/problems/reverse-linked-list/)

# 24. 两两交换链表中的节点

[力扣题目链接](https://leetcode.cn/problems/swap-nodes-in-pairs/)
## 思路

这道题目正常模拟就可以了。

建议使用虚拟头结点，这样会方便很多，要不然每次针对头结点（没有前一个指针指向头结点），还要单独处理。

# 19.删除链表的倒数第N个节点

[力扣题目链接](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)
进阶：你能尝试使用一趟扫描实现吗？
## 思路


双指针的经典应用，如果要删除倒数第n个节点，让fast移动n步，然后让fast和slow同时移动，直到fast指向链表末尾。删掉slow所指向的节点就可以了。
推荐使用虚拟头结点，这样方便处理删除实际头结点的逻辑。


# 面试题 02.07. 链表相交 

同：160.链表相交

[力扣题目链接](https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/)
# 142.环形链表II

[力扣题目链接](https://leetcode.cn/problems/linked-list-cycle-ii/)

## 思路

主要考察两知识点：

* 判断链表是否环
* 如果有环，如何找到这个环的入口
### 判断链表是否有环

可以使用快慢指针法，分别定义 fast 和 slow 指针，从头结点出发，fast指针每次移动两个节点，slow指针每次移动一个节点，如果 fast 和 slow指针在途中相遇 ，说明这个链表有环。


### 如果有环，如何找到这个环的入口

**此时已经可以判断链表是否有环了，那么接下来要找这个环的入口了。**

假设从头结点到环形入口节点 的节点数为x。
环形入口节点到 fast指针与slow指针相遇节点 节点数为y。
从相遇节点  再到环形入口节点节点数为 z。 如图所示：

![](https://code-thinking-1253855093.file.myqcloud.com/pics/20220925103433.png)

那么相遇时：
slow指针走过的节点数为: `x + y`，
fast指针走过的节点数：` x + y + n (y + z)`，n为fast指针在环内走了n圈才遇到slow指针， （y+z）为 一圈内节点的个数A。

因为fast指针是一步走两个节点，slow指针一步走一个节点， 所以 fast指针走过的节点数 = slow指针走过的节点数 * 2：

`(x + y) * 2 = x + y + n (y + z)`

两边消掉一个（x+y）: `x + y  = n (y + z) `

因为要找环形的入口，那么要求的是x，因为x表示 头结点到 环形入口节点的的距离。

所以要求x ，将x单独放在左面：`x = n (y + z) - y` ,

再从n(y+z)中提出一个 （y+z）来，整理公式之后为如下公式：`x = (n - 1) (y + z) + z  ` 注意这里n一定是大于等于1的，因为 fast指针至少要多走一圈才能相遇slow指针。

这个公式说明什么呢？

先拿n为1的情况来举例，意味着fast指针在环形里转了一圈之后，就遇到了 slow指针了。

当 n为1的时候，公式就化解为 `x = z`，

这就意味着，**从头结点出发一个指针，从相遇节点 也出发一个指针，这两个指针每次只走一个节点， 那么当这两个指针相遇的时候就是 环形入口的节点**。


也就是在相遇节点处，定义一个指针index1，在头结点处定一个指针index2。

让index1和index2同时移动，每次移动一个节点， 那么他们相遇的地方就是 环形入口的节点。
那么 n如果大于1是什么情况呢，就是fast指针在环形转n圈之后才遇到 slow指针。

其实这种情况和n为1的时候 效果是一样的，一样可以通过这个方法找到 环形的入口节点，只不过，index1 指针在环里 多转了(n-1)圈，然后再遇到index2，相遇点依然是环形的入口节点。
# 234.回文链表

[力扣题目链接](https://leetcode.cn/problems/palindrome-linked-list/)


# 143.重排链表

[力扣题目链接](https://leetcode.cn/problems/reorder-list/submissions/)

