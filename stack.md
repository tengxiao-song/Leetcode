# 栈与队列理论基础

我想栈和队列的原理大家应该很熟悉了，队列是先进先出，栈是先进后出。

如图所示：

![栈与队列理论1](https://code-thinking-1253855093.file.myqcloud.com/pics/20210104235346563.png) 

# 232.用栈实现队列

[力扣题目链接](https://leetcode.cn/problems/implement-queue-using-stacks/) 
 
* 你只能使用标准的栈操作 -- 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。 (只能用pop，不能用pop(0),res[0])
  
使用栈来模式队列的行为，如果仅仅用一个栈，是一定不行的，所以需要两个栈**一个输入栈，一个输出栈**，这里要注意输入栈和输出栈的关系。
 
在push数据的时候，只要数据放进输入栈就好，**但在pop的时候，操作就复杂一些，输出栈如果为空，就把进栈数据全部导入进来（注意是全部导入）**，再从出栈弹出数据，如果输出栈不为空，则直接从出栈弹出数据就可以了。

# 225. 用队列实现栈

[力扣题目链接](https://leetcode.cn/problems/implement-stack-using-queues/)

只能用popleft(), 不能用res[-1]

优化，使用一个队列实现
```
class MyStack:

    def __init__(self):
        self.que = deque()

    def push(self, x: int) -> None:
        self.que.append(x)

    def pop(self) -> int:
        for i in range(len(self.que)-1):
            self.que.append(self.que.popleft())
        return self.que.popleft()

    def top(self) -> int:
        for i in range(len(self.que)-1):
            self.que.append(self.que.popleft())
        temp = self.que.popleft()
        self.que.append(temp)
        return temp

    def empty(self) -> bool:
        return not self.que
```

# 20. 有效的括号

[力扣题目链接](https://leetcode.cn/problems/valid-parentheses/)

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

