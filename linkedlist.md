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
