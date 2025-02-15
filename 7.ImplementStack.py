# Write a Python program to implement a stack using lists. A stack follows the Last 
# In First Out (LIFO) principle, where elements are added and removed from the same end.

class Stack:
    def __init__(self):
        self.stack=[]
    
    def push(self,item):
        self.stack.append(item)
    
    def pop(self):
        if self.is_empty():
            return None
        return self.stack.pop()
    def peek(self):
        if self.is_empty():
            return None
        return self.stack[-1]
    
    def is_empty(self):
        return len(self.stack)==0
    
    def size(self):
        return len(self.stack)
    

stack= Stack()
stack.push(10)
stack.push(30)
stack.push(50)
print(stack.pop())
print(stack.peek())
print(stack.size())