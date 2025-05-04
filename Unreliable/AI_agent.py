import heapq
from itertools import permutations,product
from collections import deque

class AI_Agent:
    def __init__(self, grid_size, start, goal, topology="moore"):
        self.grid_size = grid_size
        self.start = tuple(start)  # Initial block positions
        self.goal = set(tuple(g) for g in goal)  # Goal is a set (any order allowed)
        self.topology = topology
        self.moves = self.get_moves()

    def get_moves(self):
        """Returns valid move directions based on topology (Moore/Von Neumann)."""
        if self.topology == "moore":
            return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return [(-1, 0), (0, -1), (0, 1), (1, 0)]  # Von Neumann

    def heuristic(self, positions):
        """ 
        Computes heuristic by summing each block's Manhattan distance 
        to its closest available goal position.
        """
        positions = list(positions)
        goal_list = list(self.goal)
        total_cost = 0

        for pos in positions:
            min_dist = min(abs(pos[0] - gx) + abs(pos[1] - gy) for gx, gy in goal_list)
            total_cost += min_dist

        return total_cost

    def is_valid(self, positions):
        """Ensures positions are within grid bounds and don't overlap."""
        return (
            all(0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] for x, y in positions)
            and len(set(positions)) == len(positions)  # Ensure no overlapping blocks
        )

    def get_neighbors(self, current):
        """
        Generates all possible next states where blocks can move in any order
        while remaining connected.
        """
        neighbors = []
        move_combinations = product(self.moves, repeat=len(current))

        for move_set in move_combinations:
            new_positions = tuple(sorted((x + dx, y + dy) for (x, y), (dx, dy) in zip(current, move_set)))

        # Check if the new positions are valid
            if self.is_valid(new_positions) and self.is_connected(new_positions):
                neighbors.append(new_positions)
        return neighbors
    

    def is_within_bounds(self, pos):
        """Checks if a position is inside the grid."""
        return 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]

    def is_connected(self, positions):
        """Ensures all blocks remain connected by checking with BFS."""
        positions = set(positions)
        to_visit = {next(iter(positions))}  # Start from any block
        visited = set()

        while to_visit:
            current = to_visit.pop()
            visited.add(current)

            for dx, dy in self.moves:  # Check all adjacent positions
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in positions and neighbor not in visited:
                    to_visit.add(neighbor)

        return len(visited) == len(positions)  # All blocks must be reachable

    def search(self):
        """Performs A* search to find the optimal transformation."""
        open_list = []
        heapq.heappush(open_list, (0, self.start))
        came_from = {self.start: None}
        g_score = {self.start: 0}

        while open_list:
            _, current = heapq.heappop(open_list)

            if set(current) == self.goal:  # Goal check (any order is fine)
                return self.reconstruct_path(came_from)

            for neighbor in self.get_neighbors(current):
                temp_g_score = g_score[current] + 1  # Uniform cost per move
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    g_score[neighbor] = temp_g_score
                    f_score = temp_g_score + self.heuristic(neighbor)
                    heapq.heappush(open_list, (f_score, neighbor))
                    came_from[neighbor] = current

        return None  # No valid path found

    def reconstruct_path(self, came_from):
        """Reconstructs the path from goal to start."""
        path = []
        current = next(pos for pos in came_from if set(pos) == self.goal)  # Find final state
        while current:
            path.append(current)
            current = came_from[current]
        return path[::-1]  # Return reversed path
