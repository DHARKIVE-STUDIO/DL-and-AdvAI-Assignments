import matplotlib.pyplot as plt
import copy
import networkx as nx
import heapq
import numpy as np

def hello():
  print("Hello Deep Learning")

def um_id() -> str:
    my_id = "82653870"
    return my_id

def unique_name() -> str:
    my_unique_name = "dharshaa"
    return my_unique_name


class Stack:
    """
    A simple implementation of a Stack data structure using a Python list.

    Methods:
    --------
    push(item):
        Adds an item to the top of the stack.

    pop():
        Removes and returns the top item from the stack. If the stack is empty, returns None.

    is_empty():
        Returns True if the stack is empty, otherwise False.

    peek():
        Returns the top item from the stack without removing it. If the stack is empty, returns None.

    size():
        Returns the number of items in the stack.
    """

    def __init__(self):
        """
        Initializes an empty stack.
        """
        self.stack = []

    def push(self, item):
        """
        Adds an item to the top of the stack.

        Parameters:
        -----------
        item : any
            The item to be added to the stack.

        Returns:
        --------
        None
        """
        ##############################################################################
        #                    TODO: Write you code                                    #
        ##############################################################################
        
        ##############################################################################
        #                     END OF YOUR CODE                                       #
        ##############################################################################
        self.stack.append(item)

    def pop(self):
        """
        Removes and returns the top item from the stack.

        Returns:
        --------
        item : any
            The item removed from the top of the stack, or None if the stack is empty.
        """
        if self.stack:
            item = self.stack.pop()
            return item

    def is_empty(self):
        """
        Checks if the stack is empty.

        Returns:
        --------
        bool
            True if the stack is empty, otherwise False.
        """
        is_empty = len(self.stack) == 0  
        return is_empty


    def size(self):
        """
        Returns the number of items in the stack.

        Returns:
        --------
        int
            The number of items in the stack.
        """
        number_of_elements = len(self.stack) 
        return number_of_elements

class Queue:
    """
    A simple implementation of a Queue data structure using a Python list.

    Methods:
    --------
    push(item):
        Adds an item to the end of the queue.

    front():
        Removes and returns the front item from the queue. If the queue is empty, returns None.

    is_empty():
        Returns True if the queue is empty, otherwise False.

    size():
        Returns the number of items in the queue.
    """

    def __init__(self):
        """
        Initializes an empty queue.
        """
        self.queue = []

    def push(self, item):
        """
        Adds an item to the end of the queue.

        Parameters:
        -----------
        item : any
            The item to be added to the queue.

        Returns:
        --------
        None
        """
        self.queue.append(item)

    def front(self):
        """
        Removes and returns the front item from the queue.

        Returns:
        --------
        item : any
            The item removed from the front of the queue, or None if the queue is empty.
        """
        if self.queue:
            return self.queue.pop(0)
        return None

    def is_empty(self):
        """
        Checks if the queue is empty.

        Returns:
        --------
        bool
            True if the queue is empty, otherwise False.
        """

        is_empty = len(self.queue) == 0


        return is_empty

    def size(self):
        """
        Returns the number of items in the queue.

        Returns:
        --------
        int
            The number of items in the queue.
        """

        number_of_elements = len(self.queue)

        return number_of_elements

class TreeNode:
    """
    A class representing a node in a tree.

    Methods:
    --------
    add_child(child):
        Adds a child node to this node.
    
    get_children():
        Returns the list of children of this node.
    """
    def __init__(self, value):
        """
        Initializes a tree node with a given value.
        
        Parameters:
        -----------
        value : any
            The value to store in the node.
        """
        self.value = value
        self.depth = 0
        self.children = []

    def get_value(self):
      return self.value

    def add_child(self, child):
        """
        Adds a child to this node.

        Parameters:
        -----------
        child : TreeNode
            The child node to be added.
        """
        child.depth = self.depth + 1
        self.children.append(child)

    def get_children(self):
        """
        Returns the list of children of this node.

        Returns:
        --------
        list of TreeNode
            The list of child nodes.
        """
        ##############################################################################
        #                    TODO: Write your code                                   #
        ##############################################################################
        children = self.children
        ##############################################################################
        #                     END OF YOUR CODE                                       #
        ##############################################################################
        return children

def dfs_search(root, target):
    """
    Performs DFS with action tracking on the tree using a stack.

    Parameters:
    -----------
    root : TreeNode
        The root node of the tree.
    target : any
        The value to search for.

    Returns:
    --------
    TreeNode or None
        The node containing the target value, or None if not found.
    """
    stack = Stack()
    stack.push(root)

    while not stack.is_empty():
        current_node = stack.pop()
        
        if current_node.get_value() == target:
            return current_node  # Return the node when found

        for child in reversed(current_node.get_children()):  # Process children in order
            stack.push(child)

    return None

def bfs_search(root, target):
    """
    Performs BFS with action tracking on the tree using a queue.

    Parameters:
    -----------
    root : TreeNode
        The root node of the tree.
    target : any
        The value to search for.

    Returns:
    --------
    TreeNode or None
        The node containing the target value, or None if not found.
    """
    queue = Queue()
    queue.push(root)

    while not queue.is_empty():
        current_node = queue.front()  # Get the first node in the queue
        
        if current_node.get_value() == target:
            return current_node  # Target found, return the node

        for child in current_node.get_children():  # Add children to queue
            queue.push(child)

    return None

from collections import deque

def flip(pancakes, k):
    """
    Reverses the order of the first k pancakes in the list.

    Parameters:
    -----------
    pancakes : list of int
        The list of pancakes.
    k : int
        The number of pancakes from the top to flip.

    Returns:
    --------
    list of int
        The pancake list after flipping the top k pancakes.
    """
    flipped_pancakes = pancakes[:k][::-1] + pancakes[k:]
    return flipped_pancakes

def minimal_flips_to_sort_pancakes(pancake_order):
    """
    Finds the minimal number of flips required to sort the pancakes using BFS.

    Parameters:
    -----------
    pancake_order : list of int
        A list representing the size of each pancake in the stack.

    Returns:
    --------
    int
        The minimal number of flips to sort the pancakes.
    """

    goal_state = tuple(sorted(pancake_order))  # Sorted pancake stack (goal state)
    initial_state = tuple(pancake_order)

    if initial_state == goal_state:
        return 0  # Already sorted, no flips needed

    queue = deque([(initial_state, 0)])  # (pancake state, flip count)
    visited = set()
    visited.add(initial_state)

    while queue:
        current_state, flips = queue.popleft()

        for k in range(2, len(current_state) + 1):
            new_state = flip(list(current_state), k)  # Flip top `k` pancakes
            new_state_tuple = tuple(new_state)

            if new_state_tuple == goal_state:
                return flips + 1  # Found the sorted state, return flips count

            if new_state_tuple not in visited:
                visited.add(new_state_tuple)
                queue.append((new_state_tuple, flips + 1))

    return -1  # This should never happen for a valid pancake sorting problem



class GraphNode:
    """
    A class representing a node in a graph.

    Methods:
    --------
    add_edge(neighbor, cost=0):
        Adds an edge between this node and a neighboring node with a specified cost.
    
    get_neighbors():
        Returns the list of neighbors connected to this node.
    """
    def __init__(self, value):
        """
        Initializes a graph node with a given value.
        
        Parameters:
        -----------
        value : any
            The value to store in the node.
        """
        self.value = value
        self.edges = []  # Stores tuples of (neighbor, edge_cost)

    def get_value(self):
        """
        Returns the value of the node.
        
        Returns:
        --------
        value : any
            The value stored in this node.
        """
        return self.value

    def add_edge(self, neighbor, cost=0):
        """
        Adds an edge to a neighboring node with a specified cost.

        Parameters:
        -----------
        neighbor : GraphNode
            The neighbor node to be connected.
        cost : int, optional
            The cost associated with the edge (default is 0).
        """
        self.edges.append((neighbor, cost))

    def get_neighbors(self):
        """
        Returns the list of neighbors connected to this node.

        Returns:
        --------
        list of tuples
            Each tuple contains (neighbor node, edge cost).
        """
        return self.edges

import heapq

def dijkstra(start_node, target_node):
    """
    Finds the shortest path from start_node to target_node using Dijkstra's algorithm.

    Parameters:
    -----------
    start_node : GraphNode
        The starting node for the algorithm.
    target_node : GraphNode
        The node to which the shortest path is calculated.

    Returns:
    --------
    tuple
        The shortest distance to the target_node and the path taken to reach it.
        If there is no path, return (-1, []).
    """
    priority_queue = []
    heapq.heappush(priority_queue, (0, id(start_node), start_node))  # (distance, unique_id, node)
    
    distances = {start_node: 0}
    previous_nodes = {start_node: None}

    while priority_queue:
        current_distance, _, current_node = heapq.heappop(priority_queue)

        if current_node == target_node:
            break  # Stop when we reach the target

        for neighbor, cost in current_node.get_neighbors():
            distance = current_distance + cost
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, id(neighbor), neighbor))  # Use id() for uniqueness

    # If target node was never reached, return -1 and an empty path
    if target_node not in distances:
        return -1, []

    # **Reconstruct the shortest path**
    path = []
    node = target_node
    while node is not None:
        path.append(node)
        node = previous_nodes.get(node)

    path.reverse()  # Reverse to get correct order

    return distances[target_node], path



class GridWorld:
    def __init__(self, height, width):
        """
        Initialize the grid world with given width and height.
        """
        self.grid = np.zeros((height, width), dtype=int)  # 0 represents free space, 1 represents obstacle
    
    def add_obstacle(self, position):
        """
        Adds an obstacle at the specified position.
        """
        if self._is_valid_position(position):
            self.grid[position] = 1  # Set obstacle
        else:
            raise ValueError("Position out of bounds")
    
    def clear_obstacle(self, position):
        """
        Clears an obstacle at the specified position.
        """
        if self._is_valid_position(position):
            self.grid[position] = 0  # Clear obstacle
        else:
            raise ValueError("Position out of bounds")
    
    def get_neighbors(self, position):
        """
        Get valid neighbors (4-directional movement: up, down, left, right)
        Returns positions that are not blocked by obstacles.
        """
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for d in directions:
            new_pos = (position[0] + d[0], position[1] + d[1])
            if self._is_valid_position(new_pos) and self.grid[new_pos] == 0:  # Not an obstacle
                neighbors.append(new_pos)
        return neighbors
    
    def _is_valid_position(self, position):
        """
        Check if the position is within the grid bounds.
        """
        return 0 <= position[0] < self.grid.shape[0] and 0 <= position[1] < self.grid.shape[1]


# Heuristic: Manhattan distance between the current node and the goal
import heapq

def astar(grid_world, start, goal):
    """
    A* pathfinding algorithm to find the shortest path in a grid world.

    Args:
        grid_world (GridWorld): The grid world object containing the grid and methods.
        start (tuple): The start position as a tuple of (row, col).
        goal (tuple): The goal position as a tuple of (row, col).

    Returns:
        list: A list of tuples representing the path from start to goal (including both),
              or an empty list if no path is found.
    """
    
    def heuristic(node, goal):
        """Manhattan distance heuristic."""
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    # Priority queue (min-heap) for A* search
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start, None))  # (f, g, node, parent)

    came_from = {}
    g_score = {start: 0}

    while open_list:
        _, g, current, parent = heapq.heappop(open_list)

        if current in came_from:  # Skip if already visited
            continue

        came_from[current] = parent  # Track where we came from

        if current == goal:  # Goal reached, reconstruct path
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Reverse the path to get the correct order

        for neighbor in grid_world.get_neighbors(current):
            temp_g_score = g + 1  # Each step cost is 1
            if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                g_score[neighbor] = temp_g_score
                f_score = temp_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score, temp_g_score, neighbor, current))

    return []  # Return an empty path if no valid route is found


##################################################################
# Functions below are all provided. You don't need to implement anything.
##################################################################
##################################################################

def plot_tree(node, x=0, y=0, x_offset=1, y_offset=-1, level=0, parent_coords=None, text_x_offset=0, text_y_offset=0):
    """
    Recursively plots the tree structure using matplotlib.
    
    Parameters:
    -----------
    node : TreeNode
        The root or current node of the tree.
    x, y : float
        The coordinates of the current node.
    x_offset : float
        The horizontal offset between nodes at the same level.
    y_offset : float
        The vertical offset between levels of the tree.
    level : int
        The current level (depth) of the node in the tree.
    parent_coords : tuple or None
        The coordinates of the parent node (used for drawing lines).
    text_x_offset : float
        The horizontal offset for the node label text.
    text_y_offset : float
        The vertical offset for the node label text.
    """
    if node is None:
        return

    # Plot the current node as a point
    plt.scatter(x, y, s=100, color='blue', zorder=5)

    # Plot the node's value with an offset for the text position
    plt.text(x + text_x_offset, y + text_y_offset, f'{node.value}', fontsize=12, ha='center', zorder=6)

    # If there's a parent node, draw a line from the parent to the current node
    if parent_coords:
        plt.plot([parent_coords[0], x], [parent_coords[1], y], color='black', zorder=4)

    # Calculate the new positions for the children nodes
    num_children = len(node.get_children())
    if num_children > 0:
        # Determine the starting x position for the first child
        x_start = x - (num_children - 1) * x_offset / 2
        for i, child in enumerate(node.get_children()):
            # Recursively plot each child
            plot_tree(child, x_start + i * x_offset, y + y_offset, x_offset * 0.7, y_offset, level + 1, (x, y), text_x_offset, text_y_offset)

def visualize_tree(root, text_x_offset=0, text_y_offset=0):
    """
    Sets up the plot and calls the recursive plot_tree function to visualize the tree.

    Parameters:
    -----------
    root : TreeNode
        The root node of the tree to be visualized.
    text_x_offset : float
        The horizontal offset for the node label text.
    text_y_offset : float
        The vertical offset for the node label text.
    """
    plt.figure(figsize=(10, 6))
    plot_tree(root, text_x_offset=text_x_offset, text_y_offset=text_y_offset)
    plt.title('Tree Structure')
    plt.axis('off')  # Turn off the axis
    plt.show()



def plot_graph(graph_nodes):
    """
    Plots a graph structure using networkx and matplotlib with a fixed seed to ensure consistency.
    
    Parameters:
    -----------
    graph_nodes : list of GraphNode
        A list of GraphNode objects representing the graph to be visualized.
    """
    G = nx.DiGraph()  # Create a directed graph

    # Add nodes to the graph
    for node in graph_nodes:
        G.add_node(node.get_value())
    
    # Add edges to the graph (with weights)
    for node in graph_nodes:
        for neighbor, cost in node.get_neighbors():
            G.add_edge(node.get_value(), neighbor.get_value(), weight=cost)

    # Get edge labels (costs)
    edge_labels = {(node.get_value(), neighbor.get_value()): cost for node in graph_nodes for neighbor, cost in node.get_neighbors()}

    # Choose a layout for the graph with a fixed seed to make the plot reproducible
    pos = nx.spring_layout(G, seed=42)  # Set a seed for consistent layout

    # Draw the nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=12, font_weight="bold", arrows=True)
    
    # Draw the edge labels (costs)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    # Display the plot
    plt.title('Graph Structure')
    plt.show()

def visualize_graph(root_node):
    """
    Visualizes the graph starting from the root node using the plot_graph function.

    Parameters:
    -----------
    root_node : GraphNode
        The root node of the graph.
    """
    # Collect all nodes for the graph by performing a DFS traversal starting from the root node
    visited = set()
    nodes = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        nodes.append(node)
        for neighbor, _ in node.get_neighbors():
            dfs(neighbor)

    dfs(root_node)

    # Plot the collected graph nodes
    plot_graph(nodes)

def visualize_grid_world(grid_world, start, goal, path=[]):
    grid = grid_world.grid
    grid_size = grid.shape
    # Visualize the grid world
    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap='gray_r')  # Obstacles will be black, free space white
    plt.scatter(start[1], start[0], c='green', s=100, label="Start")  # Start is green
    plt.scatter(goal[1], goal[0], c='red', s=100, label="Goal")  # Goal is red
    # Plot the path, if found
    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], c='blue', linewidth=2, label="Path")
    # Adding grid lines
    plt.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-0.5, grid_size[1], 1), [])
    plt.yticks(np.arange(-0.5, grid_size[0], 1), [])

    # Show labels and title
    plt.legend()
    plt.title("Grid World for A* Algorithm")
    plt.show()
