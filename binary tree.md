# 二叉树理论基础篇
## 二叉树的种类

### 满二叉树

满二叉树：如果一棵二叉树只有度为0的结点和度为2的结点，并且度为0的结点在同一层上，则这棵二叉树为满二叉树。

<img src='https://code-thinking-1253855093.file.myqcloud.com/pics/20200806185805576.png' width=600> </img></div>

这棵二叉树为满二叉树，也可以说深度为k，有2^k-1个节点的二叉树。

### 完全二叉树

完全二叉树的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层（h从1开始），则该层包含 1~ 2^(h-1)  个节点。

<img src='https://code-thinking-1253855093.file.myqcloud.com/pics/20200920221638903.png' width=600> </img></div>

### 二叉搜索树

前面介绍的树，都没有数值的，而二叉搜索树是有数值的了，**二叉搜索树是一个有序树**。


* 若它的左子树不空，则左子树上所有结点的值均小于它的根结点的值；
* 若它的右子树不空，则右子树上所有结点的值均大于它的根结点的值；
* 它的左、右子树也分别为二叉排序树

下面这两棵树都是搜索树

<img src='https://code-thinking-1253855093.file.myqcloud.com/pics/20200806190304693.png' width=600> </img></div>

### 平衡二叉搜索树

平衡二叉搜索树：又被称为AVL（Adelson-Velsky and Landis）树，且具有以下性质：它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树。

如图：

<img src='https://code-thinking-1253855093.file.myqcloud.com/pics/20200806190511967.png' width=600> </img></div>

最后一棵 不是平衡二叉树，因为它的左右两个子树的高度差的绝对值超过了1。

## 二叉树的存储方式

**二叉树可以链式存储，也可以顺序存储。**

那么链式存储方式就用指针， 顺序存储的方式就是用数组。

顾名思义就是顺序存储的元素在内存是连续分布的，而链式存储则是通过指针把分布在各个地址的节点串联一起。

链式存储如图：

<img src='https://code-thinking-1253855093.file.myqcloud.com/pics/2020092019554618.png' width=600> </img></div>

链式存储是大家很熟悉的一种方式，那么我们来看看如何顺序存储呢？

其实就是用数组来存储二叉树，顺序存储的方式如图：

<img src='https://code-thinking-1253855093.file.myqcloud.com/pics/20200920200429452.png' width=600> </img></div>

用数组来存储二叉树如何遍历的呢？

**如果父节点的数组下标是 i，那么它的左孩子就是 i * 2 + 1，右孩子就是 i * 2 + 2。**

## 二叉树的定义

刚刚我们说过了二叉树有两种存储方式顺序存储，和链式存储，顺序存储就是用数组来存，这个定义没啥可说的，我们来看看链式存储的二叉树节点的定义方式。

C++代码如下：

```cpp
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
```
# 二叉树的递归遍历

**递归三部曲**

1. **确定递归函数的参数和返回值：**
确定哪些参数是递归的过程中需要处理的，那么就在递归函数里加上这个参数， 并且还要明确每次递归的返回值是什么进而确定递归函数的返回类型。

2. **确定终止条件：**
写完了递归算法,  运行的时候，经常会遇到栈溢出的错误，就是没写终止条件或者终止条件写的不对，操作系统也是用一个栈的结构来保存每一层递归的信息，如果递归没有终止，操作系统的内存栈必然就会溢出。

3. **确定单层递归的逻辑：**
确定每一层递归需要处理的信息。在这里也就会重复调用自己来实现递归的过程。

* [144.二叉树的前序遍历](https://leetcode.cn/problems/binary-tree-preorder-traversal/)
* [145.二叉树的后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal/)
* [94.二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

也可以使用**非递归遍历**，使用stack

```python
# 前序遍历-迭代-LC144_二叉树的前序遍历
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        # 根结点为空则返回空列表
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            # 中结点先处理
            result.append(node.val)
            # 右孩子先入栈，因为要用到stack的pop
            if node.right:
                stack.append(node.right)
            # 左孩子后入栈
            if node.left:
                stack.append(node.left)
        return result
```
# 二叉树层序遍历

## 算法公开课

* [102.二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)
* [107.二叉树的层次遍历II](https://leetcode.cn/problems/binary-tree-level-order-traversal-ii/)
* [199.二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)
* [637.二叉树的层平均值](https://leetcode.cn/problems/average-of-levels-in-binary-tree/)
* [429.N叉树的层序遍历](https://leetcode.cn/problems/n-ary-tree-level-order-traversal/)
* [515.在每个树行中找最大值](https://leetcode.cn/problems/find-largest-value-in-each-tree-row/)
* [116.填充每个节点的下一个右侧节点指针](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/)
* [117.填充每个节点的下一个右侧节点指针II](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node-ii/)
* [104.二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)
* [111.二叉树的最小深度](https://leetcode.cn/problems/minimum-depth-of-binary-tree/)

# 226.翻转二叉树

[力扣题目链接](https://leetcode.cn/problems/invert-binary-tree/)

# 101. 对称二叉树

[力扣题目链接](https://leetcode.cn/problems/symmetric-tree/)

要用后续遍历，因为需要先收集孩子的信息再处理当前元素。

```py
class Solution(object):
    def isSymmetric(self, root):
        def dfs(left,right):
            if not left and not right:
                return True
            elif (not left or not right) or (left.val != right.val):
                return False
            outside = dfs(left.left, right.right)    # 左
            inside = dfs(left.right, right.left)    # 右
            return outside and inside                # 中
        return dfs(root.left, root.right)
```

# 104.二叉树的最大深度

* [104.二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)
* [559.n叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-n-ary-tree/)

本题可以使用前序（中左右），也可以使用后序遍历（左右中），使用前序求的就是深度，使用后序求的是高度。
* 二叉树节点的深度：指从根节点到该节点的最长简单路径边的条数或者节点数（取决于深度从0开始还是从1开始）
* 二叉树节点的高度：指从该节点到叶子节点的最长简单路径边的条数或者节点数（取决于高度从0开始还是从1开始）

二叉树的最大深度就是这棵树的高度，可以用后续遍历。（单求深度要用前序遍历）

```py
class Solution(object):
    def maxDepth(self, root):
        def dfs(root):
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            return max(left,right) + 1
        return dfs(root)
```
```py
class Solution(object):
    def maxDepth(self, root):
        def dfs(root):
            if not root:
                return 0
            high = 0
            for child in root.children:
                high = max(high, dfs(child))
            return high + 1
        return dfs(root)
        
```

# 111.二叉树的最小深度

[力扣题目链接](https://leetcode.cn/problems/minimum-depth-of-binary-tree/)

后续遍历 + 注意找的是根结点到**叶子**的最短深度
```py
class Solution(object):
    def minDepth(self, root):
        def dfs(root):
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            if root.left and not root.right:
                return left + 1
            if not root.left and root.right:
                return right + 1
            return min(left,right) + 1
        return dfs(root)
```

# 222.完全二叉树的节点个数(小于O(n))

[力扣题目链接](https://leetcode.cn/problems/count-complete-tree-nodes/)

完全二叉树只有两种情况，情况一：就是满二叉树，情况二：最后一层叶子节点没有满。

对于情况一，可以直接用 2^树深度 - 1 来计算，注意这里根节点深度为1。

对于情况二，分别递归左孩子，和右孩子，递归到某一深度一定会有左孩子或者右孩子为满二叉树，然后依然可以按照情况1来计算。

```py
class Solution(object):
    def countNodes(self, root):
        def dfs(root):
            if not root:
                return 0
            left_ptr = root.left
            right_ptr = root.right
            left_level = 1
            right_level = 1
            # 记录最左边高度和最右边高度，如果一样说明中间node都是满的，直接使用公示即可，省去了遍历中间节点。
            while left_ptr:
                left_ptr = left_ptr.left
                left_level += 1
            while right_ptr:
                right_ptr = right_ptr.right
                right_level += 1
            if left_level == right_level:
                return 2**left_level-1
            left_num = dfs(root.left)
            right_num = dfs(root.right)
            return left_num + right_num + 1
        return dfs(root)
```

# 110.平衡二叉树

[力扣题目链接](https://leetcode.cn/problems/balanced-binary-tree/)

分别求出其左右子树的高度，然后如果差值小于等于1，则返回当前二叉树的高度，否则返回-1，表示已经不是二叉平衡树了。

用后序遍历，这样如果有子树返回-1，当前节点也可以返回-1.

每次递归就可以传达两个信息，一个是子树是否平衡，如果平衡获得高度
```py
def isBalanced(self, root):
        def dfs(root):
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            if left == -1 or right == -1 or abs(left-right) > 1:
                return -1
            return max(left,right)+1
        return dfs(root) != -1
```


# 257. 二叉树的所有路径

[力扣题目链接](https://leetcode.cn/problems/binary-tree-paths/)

递归中还有回溯（已经收集到了125 所以需要弹出左子树25 再去找右子树的3）


# 404.左叶子之和

[力扣题目链接](https://leetcode.cn/problems/sum-of-left-leaves/)


# 513.找树左下角的值 (用递归）

[力扣题目链接](https://leetcode.cn/problems/find-bottom-left-tree-value/)

层序可以很直白，简单的求出。这里用递归+回溯练习。

```py
class Solution(object):
    def findBottomLeftValue(self, root):
        # 定义两个全局变量，持续更新
        self.max_depth = float("-inf")
        self.res = 0
        def dfs(root,depth):
            if not root:
                return
            # 如果是叶子且深度更大就更新变量，因为一直是先处理左子树再处理右子树，可以确保更新的leftmost value
            if not root.left and not root.right:
                if depth > self.max_depth:
                    self.max_depth = depth
                    self.res = root.val
            depth += 1
            dfs(root.left,depth)
            dfs(root.right,depth)
            depth -= 1    # 回溯
        dfs(root, 0)
        return self.res
```

# 112. 路径总和

[力扣题目链接](https://leetcode.cn/problems/path-sum/)

全局变量和array需要回溯，如果传参的话可以不用。

```py
class Solution(object):
    def hasPathSum(self, root, targetSum):
        self.path = 0
        def dfs(root):
            if not root:
                return False
            self.path += root.val
            # 如果为叶子结点，判断总和是否是target，如果是就不用管后续直接返回True，如果不是需要回溯再返回False
            if not root.left and not root.right:
                if self.path == targetSum:
                    return True
                else:
                    self.path -= root.val
                    return False
            # 如果不是叶子结点，就获取左右子树的返回值，回溯并返回
            left = dfs(root.left)
            right = dfs(root.right)
            self.path -= root.val
            return left or right
        return dfs(root)
```
# 106.从中序与后序遍历序列构造二叉树

[力扣题目链接](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

# 654.最大二叉树

[力扣题目地址](https://leetcode.cn/problems/maximum-binary-tree/)

# 617.合并二叉树

[力扣题目链接](https://leetcode.cn/problems/merge-two-binary-trees/)

# 700.二叉搜索树中的搜索

[力扣题目地址](https://leetcode.cn/problems/search-in-a-binary-search-tree/)

# 98.验证二叉搜索树

[力扣题目链接](https://leetcode.cn/problems/validate-binary-search-tree/)

可以用数组，或者直接搜索。这里是模拟中序搜索，先更新左节点的值，然后是中最后是后。
```
class Solution:
    def __init__(self):
        self.pre = None  # 用来记录前一个节点

    def isValidBST(self, root):
        if root is None:
            return True

        left = self.isValidBST(root.left)

        if self.pre is not None and self.pre.val >= root.val:
            return False
        self.pre = root  # 记录前一个节点

        right = self.isValidBST(root.right)
        return left and right
```
# 530.二叉搜索树的最小绝对差

[力扣题目链接](https://leetcode.cn/problems/minimum-absolute-difference-in-bst/)

# 501.二叉搜索树中的众数 (constant space/no hashmap)

[力扣题目链接](https://leetcode.cn/problems/find-mode-in-binary-search-tree/)
```
class Solution(object):
    def findMode(self, root):
        self.count = 0
        self.maxcount = 0
        self.res = []
        self.pre = None
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            if self.pre == None:
                self.count += 1     #case: first node
            elif self.pre.val == root.val:
                self.count += 1     #case:same node
            else:
                self.count = 1     #case:newnode
            self.pre = root         #update node
            if self.count == self.maxcount:
                self.res.append(root.val)
            elif self.count > self.maxcount:
                self.maxcount = self.count
                self.res = [root.val]       #update result,erase previous
            dfs(root.right)
            return
        dfs(root)
        return self.res
```


# 236. 二叉树的最近公共祖先 (又错)

[力扣题目链接](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)
```
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left,p,q)
        right = self.lowestCommonAncestor(root.right,p,q)
        if not left:return right    #case:都在右
        if not right: return left   #case:都在左
        return root         #case:左右各有一个
```
# 235. 二叉搜索树的最近公共祖先

[力扣题目链接](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/)

# 701.二叉搜索树中的插入操作

[力扣题目链接](https://leetcode.cn/problems/insert-into-a-binary-search-tree/)

# 450.删除二叉搜索树中的节点(错)

[力扣题目链接]( https://leetcode.cn/problems/delete-node-in-a-bst/)

```
class Solution:
    def deleteNode(self, root, key):
        if root is None:  # 如果根节点为空，直接返回
            return root
        if root.val == key:  # 找到要删除的节点
            if root.right is None:  # 如果右子树为空，直接返回左子树作为新的根节点
                return root.left
            cur = root.right
            while cur.left:  # 找到右子树中的最左节点
                cur = cur.left
            root.val, cur.val = cur.val, root.val  # 将要删除的节点值与最左节点值交换,这样做到删除节点时必定返回None
        root.left = self.deleteNode(root.left, key)  # 在左子树中递归删除目标节点
        root.right = self.deleteNode(root.right, key)  # 在右子树中递归删除目标节点
        return root
```


# 669. 修剪二叉搜索树(错)

[力扣题目链接](https://leetcode.cn/problems/trim-a-binary-search-tree/)

对根结点进行深度优先遍历。对于当前访问的结点，如果结点为空结点，直接返回空结点；如果结点的值小于low，那么说明该结点及它的左子树都不符合要求，我们返回对它的右结点进行修剪后的结果；如果结点的值大于high，那么说明该结点及它的右子树都不符合要求，我们返回对它的左子树进行修剪后的结果；如果结点的值位于区间，我们将结点的左结点设为对它的左子树修剪后的结果，右结点设为对它的右子树进行修剪后的结果。
```
class Solution(object):
    def trimBST(self, root, low, high):
        """
        :type root: TreeNode
        :type low: int
        :type high: int
        :rtype: TreeNode
        """
        if not root: return root
        if root.val < low:
            return self.trimBST(root.right,low,high)
        if root.val > high:
            return self.trimBST(root.left,low,high)
        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right,low,high)
        return root
```

# 108.将有序数组转换为二叉搜索树

[力扣题目链接](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)

# 538.把二叉搜索树转换为累加树(优解)

[力扣题目链接](https://leetcode.cn/problems/convert-bst-to-greater-tree/)

```
class Solution(object):
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        self.sum = 0
        def dfs(root):
            if not root:
                return
            dfs(root.right)
            self.sum += root.val
            root.val = self.sum
            dfs(root.left)
            return root
        return dfs(root)
```

# 129. 求根节点到叶节点数字之和

[力扣题目链接](https://leetcode.cn/problems/sum-root-to-leaf-numbers/)

# 1382.将二叉搜索树变平衡

[力扣题目链接](https://leetcode.cn/problems/balance-a-binary-search-tree/)

# 100. 相同的树

[力扣题目链接](https://leetcode.cn/problems/same-tree/)


# 116. 填充每个节点的下一个右侧节点指针

[力扣题目链接](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/)

