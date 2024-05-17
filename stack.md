# 栈与队列理论基础

我想栈和队列的原理大家应该很熟悉了，队列是先进先出，栈是先进后出。

如图所示：

![栈与队列理论1](https://code-thinking-1253855093.file.myqcloud.com/pics/20210104235346563.png) 

# 232.用栈实现队列

[力扣题目链接](https://leetcode.cn/problems/implement-queue-using-stacks/) 
 
* 你只能使用标准的栈操作 -- 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。 (只能用pop，不能用pop(0),res[0])
  
使用栈来模式队列的行为，如果仅仅用一个栈，是一定不行的，所以需要两个栈**一个输入栈，一个输出栈**，这里要注意输入栈和输出栈的关系。
 
在push数据的时候，只要数据放进输入栈就好，**但在pop的时候，操作就复杂一些，输出栈如果为空，就把进栈数据全部导入进来（注意是全部导入）**，再从出栈弹出数据，如果输出栈不为空，则直接从出栈弹出数据就可以了。

```py
class MyQueue(object):

    def __init__(self):
        self.input_ = []
        self.output = []

    def push(self, x):
        self.input_.append(x)

    # 如果queue的加入顺序是1，2，3，那么queue的弹出顺序是1，2，3. input栈的元素是1，2，3, 直接用pop的话顺序是反的.
    # 我们把input的元素用pop加入output，output的元素是3，2，1，再使用pop得到正确的顺序（负负得正）
    def pop(self):
        if self.output:
            return self.output.pop()
        else:
            for i in range(len(self.input_)-1):
                self.output.append(self.input_.pop())
            return self.input_.pop()

    def peek(self):
        res = self.pop()
        self.output.append(res)
        return res

    def empty(self):
        return not (self.output or self.input_)
```
# 225. 用队列实现栈

[力扣题目链接](https://leetcode.cn/problems/implement-stack-using-queues/)

只能用popleft(), 不能用res[-1]

优化，使用一个队列实现
```py
class MyStack(object):

    def __init__(self):
        self.q = collections.deque()

    def push(self, x):
        self.q.append(x)

    # 重复添加前size-1的元素，然后返回处理后的第一个元素(也就是处理前的最后一个元素)
    def pop(self):
        for i in range(len(self.q)-1):
            self.q.append(self.q.popleft())
        return self.q.popleft()


    def top(self):
        res = self.pop()
        self.q.append(res)
        return res


    def empty(self):
        return not self.q
```

# 20. 有效的括号

[力扣题目链接](https://leetcode.cn/problems/valid-parentheses/)

加入右括号的方法
```py
class Solution(object):
    def isValid(self, s):
        stack = []
        for i in s:
            if i == "(":
                stack.append(")")
            elif i == "[":
                stack.append("]")
            elif i == "{":
                stack.append("}")
            # 处理右括号多了，左右括号不一致的情况
            elif not stack or stack.pop() != i:
                return False
        # 处理左括号多了的情况
        return not stack
```

# 1047. 删除字符串中的所有相邻重复项

[力扣题目链接](https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/)

# 150. 逆波兰表达式求值

[力扣题目链接](https://leetcode.cn/problems/evaluate-reverse-polish-notation/)

坑：python 的整数除法是向下取整，而不是向零取整。

python2 的除法 "/" 是整数除法， "-3 / 2 = -2" ；

python3 的地板除 "//" 是整数除法， "-3 // 2 = -2" ；

python3 的除法 "/" 是浮点除法， "-3 / 2 = -1.5" ；

对Python 的整数除法问题，可以用 int(num1 / float(num2)) 来做，即先用浮点数除法，然后取整。

无论如何，浮点数除法都会得到一个浮点数，比如 "-3 / 2.0 = 1.5" ；

此时再取整，就会得到整数部分，即 float(-1.5) = -1 。

# 239. 滑动窗口最大值(困难)

[力扣题目链接](https://leetcode.cn/problems/sliding-window-maximum/)

在线性时间复杂度内解决此题:

用deque保留元素的下标，如果新元素更大，从后往前pop掉deque的元素。保证了q[0]一定是当前窗口最大的元素。

针对滑动区间，因为保存的是下标，如果下标小于 i - k 也就是在窗口之外，从前往后pop元素直到满足要求。由于最靠前的元素一定是最大的，在满足区间内选中第一个元素即可。

# 347.前 K 个高频元素

[力扣题目链接](https://leetcode.cn/problems/top-k-frequent-elements/)

```
        import heapq
        count = Counter(nums)
        res = []
        for key, value in count.items():
            if len(res) >= k:
                if value > res[0][0]:
                    heapq.heapreplace(res,(value,key))
            else:
                heapq.heappush(res, (value,key))
        return [item[1] for item in res]
```

