from graphviz import Digraph
from collections import deque
from predictors import Neuron
from value import Value


class GraphDrawer:

    def __init__(self):
        self.node_to_id = {}
        self.dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
        
    def get_graph(self, root):
        q = deque()
        visited = set()
        q.append(root)
        visited.add(root)
        while len(q) != 0:
            cur = q.popleft()
            self.dot.node(name=self.get_id(cur), label=str(cur), shape='record')
            cur_id_with_op = self.get_id(cur) + cur.op
            if cur.op:
                self.dot.edge(cur_id_with_op, self.get_id(cur))
                self.dot.node(name=cur_id_with_op, label=cur.op)
                
            for child in cur.children:
                self.dot.edge(self.get_id(child), cur_id_with_op)
                if child not in visited:
                    q.append(child)
                    visited.add(child)
                
    def get_id(self, node):
        if node not in self.node_to_id:
            self.node_to_id[node] = len(self.node_to_id)
        return str(self.node_to_id[node])
    
    def draw(self, node):
        g = self.get_graph(node)
        return self.dot
