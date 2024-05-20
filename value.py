import math
from collections import deque

EPSILON = 1e-9

class Value:

    def __init__(self, data, children=[], op='', label=''):
        if not label:
            raise Exception(f"Label missing for node with value {data}")
        self.data = data
        self.grad = 0.0
        self.children = children
        self.op = op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        if not self.label:
            raise Exception(f"Label missing for node with value {self.data}")
        common = f'data={self.data:.4f} | grad={self.grad:.4f}'
        return f"{self.label}|{common}"
        

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f'{str(other)}')
        res = Value(self.data + other.data, [self, other], '+', self.label + '+' + other.label)
        def _backward():
            self.grad += res.grad
            other.grad += res.grad
        res._backward = _backward
        return res

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f'{str(other)}')
        res = Value(self.data * other.data, [self, other], '*', self.label + '*' + other.label)
        def _backward():
            self.grad += res.grad * other.data
            other.grad += res.grad * self.data
        res._backward = _backward
        return res
        

    def exp(self):
        val = math.exp(self.data * 1.0)
        res = Value(val, [self], 'exp', label=f'e**({self.label})')
        def _backward():
            self.grad += res.grad * val
        res._backward = _backward
        return res

    def ln(self):
        if self.data < 0:
            raise Exception("Cant find ln of negative number")
        if abs(self.data) < EPSILON:
            self.data = EPSILON
            # raise Exception("Cant take ln of 0")
        res = Value(math.log(self.data * 1.0), [self], 'ln', label=f'ln({self.label})')
        def _backward():
            self.grad += res.grad * (1 / self.data)
        res._backward = _backward
        return res
        
    def tanh(self):
        exp = math.exp(2 * self.data * 1.0)
        val = (exp - 1) / (exp + 1)
        res = Value(val, [self], 'tanh', label=f'tanh({self.label})')
        def _backward():
            self.grad += res.grad * (1 - res.data * res.data)
        res._backward = _backward
        return res

    def sigmoid(self):
        val = 1 / (1 + math.exp(-self.data))
        res = Value(val, [self], 'sigmoid', label=f'sigmoid({self.label})')
        def _backward():
            return 0

        res._backward = _backward
        return res

    def relu(self):
        val = max(0, self.data)
        res = Value(val, [self], 'relu', label=f'relu({self.label})')
        def _backward():
            return 0

        res._backward = _backward
        return res
            
    
    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f'{str(other)}')
        res = Value(self.data ** other.data, [self, other], '**', label=f'{self.label} ** {other.label}')
        def backward():
            self.grad += res.grad * (other.data * (self.data ** (other.data - 1)))
        res._backward = backward
        return res

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f'{str(other)}')
        return self * (other ** -1)


    def __sub__(self, other):
        return self + (-other)
        
    def __neg__(self):
        return self * -1


    def topo_sort(self, root):
        res = []
        node_to_num_edges = {}
        visited = set()
        def update_edges_count(n):
            for child in n.children:
                if child not in node_to_num_edges:
                    node_to_num_edges[child] = 0
                node_to_num_edges[child] += 1
            for child in n.children:
                if child not in visited:
                    visited.add(child)            
                    update_edges_count(child)
        node_to_num_edges[root] = 0
        visited.add(root)
        update_edges_count(root)
            
        q = deque()
        visited = set()
        for node in node_to_num_edges:
            if node_to_num_edges[node] == 0:
                q.append(node)
                visited.add(node)
    
        while len(q) != 0:
            node = q.popleft()
            res.append(node)
            for child in node.children:
                node_to_num_edges[child] -= 1
            for node in node_to_num_edges:
                if node not in visited and node_to_num_edges[node] == 0:
                    q.append(node)
                    visited.add(node)
        return res
    
    def backward(self):
        self.grad = 1
        ordered_nodes = self.topo_sort(self)
        for node in ordered_nodes:
            node._backward()
