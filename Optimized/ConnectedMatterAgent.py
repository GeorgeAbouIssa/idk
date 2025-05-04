import heapq
import time
import matplotlib.pyplot as plt
from collections import deque
import threading
import concurrent.futures
from functools import partial

class ConnectedMatterAgent:
    def __init__(self, grid_size, start_positions, goal_positions, topology="moore", max_simultaneous_moves=1, min_simultaneous_moves=1, obstacles=None):
        self.grid_size = grid_size
        self.start_positions = list(start_positions)
        self.goal_positions = list(goal_positions)
        self.topology = topology
        self.max_simultaneous_moves = max_simultaneous_moves
        self.min_simultaneous_moves = min(min_simultaneous_moves, max_simultaneous_moves)  # Ensure min <= max
        self.obstacles = set(obstacles) if obstacles else set()
        
        # Set moves based on topology
        if self.topology == "moore":
            self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:  # Von Neumann
            self.directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            
        # Initialize the start and goal states
        self.start_state = frozenset((x, y) for x, y in start_positions)
        self.goal_state = frozenset((x, y) for x, y in goal_positions)
        
        # New: Check if goal state is disconnected and find components
        self.goal_components = self.find_disconnected_components(self.goal_state)
        self.is_goal_disconnected = len(self.goal_components) > 1
        
        if self.is_goal_disconnected:
            print(f"Goal state has {len(self.goal_components)} disconnected components")
            # Calculate centroids for each component
            self.component_centroids = [self.calculate_centroid(comp) for comp in self.goal_components]
            # Calculate the overall goal centroid
            self.goal_centroid = self.calculate_centroid(self.goal_positions)
        else:
            # Calculate the centroid of the goal positions for block movement phase
            # Using exact position calculation instead of average to ensure precise positioning
            self.goal_centroid = self.calculate_centroid(self.goal_positions)
        
        # Cache for valid moves to avoid recomputation
        self.valid_moves_cache = {}
        
        # For optimizing the search
        self.articulation_points_cache = {}
        self.connectivity_check_cache = {}
        
        # Enhanced parameters for improved search
        self.beam_width = 800  # Increased beam width for better exploration
        self.max_iterations = 100000  # Limit iterations to prevent infinite loops
        
        # For obstacle pathfinding optimization
        self.distance_map_cache = {}
        self.obstacle_maze = None
        if obstacles:
            self.build_obstacle_maze()
            
        # NEW: Track blocks that have reached their goal positions
        self.blocks_at_goal = set()
            
    def calculate_centroid(self, positions):
        """Calculate the centroid (average position) of a set of positions"""
        if not positions:
            return (0, 0)
        x_sum = sum(pos[0] for pos in positions)
        y_sum = sum(pos[1] for pos in positions)
        return (x_sum / len(positions), y_sum / len(positions))
    
    def is_connected(self, positions):
        """Check if all positions are connected using BFS"""
        if not positions:
            return True
            
        # Use cache if available
        positions_hash = hash(frozenset(positions))
        if positions_hash in self.connectivity_check_cache:
            return self.connectivity_check_cache[positions_hash]
            
        # Convert to set for O(1) lookup
        positions_set = set(positions)
        
        # Start BFS from first position
        start = next(iter(positions_set))
        visited = {start}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            
            # Check all adjacent positions
            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in positions_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # All positions should be visited if connected
        is_connected = len(visited) == len(positions_set)
        
        # Cache the result
        self.connectivity_check_cache[positions_hash] = is_connected
        return is_connected
    
    def get_articulation_points(self, state_set):
        """
        Find articulation points (critical points that if removed would disconnect the structure)
        Uses a modified DFS algorithm
        """
        state_hash = hash(frozenset(state_set))
        if state_hash in self.articulation_points_cache:
            return self.articulation_points_cache[state_hash]
            
        if len(state_set) <= 2:  # All points are critical in structures of size 1-2
            self.articulation_points_cache[state_hash] = set(state_set)
            return set(state_set)
            
        articulation_points = set()
        visited = set()
        discovery = {}
        low = {}
        parent = {}
        time = [0]  # Using list to allow modification inside nested function
        
        def dfs(u, time):
            children = 0
            visited.add(u)
            discovery[u] = low[u] = time[0]
            time[0] += 1
            
            # Visit all neighbors
            for dx, dy in self.directions:
                v = (u[0] + dx, u[1] + dy)
                if v in state_set:
                    if v not in visited:
                        children += 1
                        parent[v] = u
                        dfs(v, time)
                        
                        # Check if subtree rooted with v has a connection to ancestors of u
                        low[u] = min(low[u], low[v])
                        
                        # u is an articulation point if:
                        # 1) u is root and has two or more children
                        # 2) u is not root and low value of one of its children >= discovery value of u
                        if parent.get(u) is None and children > 1:
                            articulation_points.add(u)
                        if parent.get(u) is not None and low[v] >= discovery[u]:
                            articulation_points.add(u)
                            
                    elif v != parent.get(u):  # Update low value of u for parent function calls
                        low[u] = min(low[u], discovery[v])
        
        # Call DFS for all vertices
        for point in state_set:
            if point not in visited:
                dfs(point, time)
                
        self.articulation_points_cache[state_hash] = articulation_points
        return articulation_points
    
    def get_valid_block_moves(self, state):
        """
        Generate valid moves for the entire block of elements
        A valid block move shifts all elements in the same direction while maintaining connectivity
        """
        valid_moves = []
        state_list = list(state)
        
        # Try moving the entire block in each direction
        for dx, dy in self.directions:
            # Calculate new positions after moving
            new_positions = [(pos[0] + dx, pos[1] + dy) for pos in state_list]
            
            # Check if all new positions are valid (within bounds, not occupied by obstacles)
            all_valid = all(0 <= pos[0] < self.grid_size[0] and 
                            0 <= pos[1] < self.grid_size[1] and
                            pos not in self.obstacles for pos in new_positions)
            
            # Only consider moves that keep all positions within bounds and not overlapping obstacles
            if all_valid:
                # NEW: Ensure no positions overlap - each position must be unique
                if len(set(new_positions)) == len(new_positions):
                    new_state = frozenset(new_positions)
                    valid_moves.append(new_state)
        
        return valid_moves
    
    def get_valid_morphing_moves(self, state):
        """
        Generate valid morphing moves that maintain connectivity
        Supports multiple simultaneous block movements with minimum requirement
        Now optimized for obstacle environments
        """
        state_key = hash(state)
        if state_key in self.valid_moves_cache:
            return self.valid_moves_cache[state_key]
            
        # Get single block moves first
        single_moves = []
        state_set = set(state)
        
        # NEW: Identify blocks that have reached their goal positions
        blocks_at_goal = state_set.intersection(self.goal_state)
        self.blocks_at_goal = blocks_at_goal
        
        # Find non-critical points that can move without breaking connectivity
        articulation_points = self.get_articulation_points(state_set)
        
        # NEW: Don't move blocks that have reached their goal positions
        # unless they're the only blocks we can move (to avoid deadlock)
        movable_points = state_set - articulation_points - blocks_at_goal
        
        # If all points are critical or at goal, try moving critical points that aren't at goals
        if not movable_points:
            for point in articulation_points - blocks_at_goal:
                # Try removing and see if structure remains connected
                temp_state = state_set.copy()
                temp_state.remove(point)
                if self.is_connected(temp_state):
                    movable_points.add(point)
                    
        # If still no movable points and we have blocks at goal,
        # allow minimal movement of goal blocks as last resort
        if not movable_points and blocks_at_goal:
            # Try moving goal blocks that aren't critical articulation points first
            non_critical_goal_blocks = blocks_at_goal - articulation_points
            if non_critical_goal_blocks:
                for point in non_critical_goal_blocks:
                    temp_state = state_set.copy()
                    temp_state.remove(point)
                    if self.is_connected(temp_state):
                        movable_points.add(point)
            
            # If still stuck, try critical goal blocks as absolute last resort
            if not movable_points:
                for point in blocks_at_goal.intersection(articulation_points):
                    temp_state = state_set.copy()
                    temp_state.remove(point)
                    if self.is_connected(temp_state):
                        movable_points.add(point)
        
        # Generate single block moves, prioritizing moves toward the goal
        for point in movable_points:
            # Find closest goal position for this point
            closest_goal = None
            min_dist = float('inf')
            
            for goal_pos in self.goal_state:
                if goal_pos not in state_set:  # Only consider unoccupied goals
                    if self.obstacles:
                        dist = self.obstacle_aware_distance(point, goal_pos)
                    else:
                        dist = abs(point[0] - goal_pos[0]) + abs(point[1] - goal_pos[1])
                        
                    if dist < min_dist:
                        min_dist = dist
                        closest_goal = goal_pos
                        
            # Prioritize directions toward the closest goal
            ordered_directions = self.directions.copy()
            if closest_goal:
                # Calculate direction vectors
                dx = 1 if closest_goal[0] > point[0] else -1 if closest_goal[0] < point[0] else 0
                dy = 1 if closest_goal[1] > point[1] else -1 if closest_goal[1] < point[1] else 0
                
                # Put preferred direction first
                preferred_dir = (dx, dy)
                if preferred_dir in ordered_directions:
                    ordered_directions.remove(preferred_dir)
                    ordered_directions.insert(0, preferred_dir)
                
                # Also prioritize partial matches (just x or just y component)
                for i, dir in enumerate(ordered_directions.copy()):
                    if dir[0] == dx or dir[1] == dy:
                        ordered_directions.remove(dir)
                        ordered_directions.insert(1, dir)
                
            # Try moving in each direction, starting with preferred ones
            for dx, dy in ordered_directions:
                new_pos = (point[0] + dx, point[1] + dy)
                
                # Skip if out of bounds or is an obstacle
                if not (0 <= new_pos[0] < self.grid_size[0] and 
                        0 <= new_pos[1] < self.grid_size[1]):
                    continue
                    
                # Skip if position is an obstacle
                if new_pos in self.obstacles:
                    continue
                
                # Skip if already occupied
                if new_pos in state_set:
                    continue
                
                # Create new state by moving the point
                new_state_set = state_set.copy()
                new_state_set.remove(point)
                new_state_set.add(new_pos)
                
                # Check if new position is adjacent to at least one other point
                has_adjacent = False
                for adj_dx, adj_dy in self.directions:
                    adj_pos = (new_pos[0] + adj_dx, new_pos[1] + adj_dy)
                    if adj_pos in new_state_set and adj_pos != new_pos:
                        has_adjacent = True
                        break
                
                # Only consider moves that maintain connectivity
                if has_adjacent and self.is_connected(new_state_set):
                    single_moves.append((point, new_pos))
                    
        # Start with empty valid moves list
        valid_moves = []
        
        # In dense obstacle environments, more simultaneous moves could be better
        # Increase minimum number of simultaneous moves based on obstacle density
        local_min_moves = self.min_simultaneous_moves
        if len(self.obstacles) > 20 and local_min_moves == 1:
            local_min_moves = min(2, self.max_simultaneous_moves)
            
        # Generate multi-block moves
        for k in range(local_min_moves, min(self.max_simultaneous_moves + 1, len(single_moves) + 1)):
            # Generate combinations of k moves
            for combo in self._generate_move_combinations(single_moves, k):
                # Check if the combination is valid (no conflicts)
                if self._is_valid_move_combination(combo, state_set):
                    # Apply the combination and check connectivity
                    new_state = self._apply_moves(state_set, combo)
                    if self.is_connected(new_state):
                        valid_moves.append(frozenset(new_state))
        
        # If no valid moves with min_simultaneous_moves, fallback to single moves if allowed
        if not valid_moves and self.min_simultaneous_moves == 1:
            valid_moves = [frozenset(self._apply_moves(state_set, [move])) for move in single_moves]
        
        # Cache results
        self.valid_moves_cache[state_key] = valid_moves
        return valid_moves
    
    def _generate_move_combinations(self, single_moves, k):
        """Generate all combinations of k moves from the list of single moves"""
        if k == 1:
            return [[move] for move in single_moves]
        
        result = []
        for i in range(len(single_moves) - k + 1):
            move = single_moves[i]
            for combo in self._generate_move_combinations(single_moves[i+1:], k-1):
                result.append([move] + combo)
        
        return result
    
    def _is_valid_move_combination(self, moves, state_set):
        """Check if a combination of moves is valid (no conflicts)"""
        # Extract source and target positions
        sources = set()
        targets = set()
        
        for src, tgt in moves:
            # Check for overlapping sources or targets
            if src in sources or tgt in targets:
                return False
            sources.add(src)
            targets.add(tgt)
            
            # Check that no target is also a source for another move
            if tgt in sources or src in targets:
                return False
                
            # NEW: Check that target doesn't overlap with any non-moving block
            non_moving_blocks = state_set - sources
            if tgt in non_moving_blocks:
                return False
        
        return True
    
    def _apply_moves(self, state_set, moves):
        """Apply a list of moves to the state"""
        new_state = state_set.copy()
        
        # NEW: First validate that we won't have any overlaps
        sources = set()
        targets = set()
        
        for src, tgt in moves:
            sources.add(src)
            targets.add(tgt)
            
        # Ensure we're not creating duplicate positions
        # 1. No target should overlap with a non-moving block
        non_moving_blocks = state_set - sources
        if targets.intersection(non_moving_blocks):
            return state_set  # Return original state if overlap detected
            
        # 2. No duplicate targets
        if len(targets) != len(moves):
            return state_set  # Return original state if duplicate targets
        
        # Apply moves only if valid
        for src, tgt in moves:
            new_state.remove(src)
            new_state.add(tgt)
            
        # Verify we haven't lost any blocks
        if len(new_state) != len(state_set):
            print(f"WARNING: Block count changed from {len(state_set)} to {len(new_state)}")
            return state_set  # Return original state if blocks were lost
            
        return new_state
    
    def get_smart_chain_moves(self, state):
        """
        Generate chain moves where one block moves into the space of another
        while that block moves elsewhere, maintaining connectivity
        Now obstacle-aware and optimized for tight spaces
        """
        state_set = set(state)
        valid_moves = []
        
        # NEW: Identify blocks that have reached their goal positions
        blocks_at_goal = state_set.intersection(self.goal_state)
        
        # For each block, try to move it toward a goal position
        for pos in state_set:
            # NEW: Skip blocks at goal positions (preserve goal state)
            if pos in blocks_at_goal:
                continue
                
            # Find closest goal position using obstacle-aware pathfinding
            min_dist = float('inf')
            closest_goal = None
            
            for goal_pos in self.goal_state:
                if goal_pos not in state_set:  # Only consider unoccupied goals
                    if self.obstacles:
                        dist = self.obstacle_aware_distance(pos, goal_pos)
                    else:
                        dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_goal = goal_pos
            
            if not closest_goal:
                continue
                
            # Calculate direction toward goal
            dx = 1 if closest_goal[0] > pos[0] else -1 if closest_goal[0] < pos[0] else 0
            dy = 1 if closest_goal[1] > pos[1] else -1 if closest_goal[1] < pos[1] else 0
            
            # Try moving in that direction
            next_pos = (pos[0] + dx, pos[1] + dy)
            
            # Skip if out of bounds or is an obstacle
            if not (0 <= next_pos[0] < self.grid_size[0] and 
                    0 <= next_pos[1] < self.grid_size[1]):
                continue
                
            # Skip if position is an obstacle
            if next_pos in self.obstacles:
                continue
            
            # If next position is occupied, try chain move
            if next_pos in state_set:
                # NEW: Skip if the blocking block is at a goal position
                if next_pos in blocks_at_goal:
                    continue
                    
                # Try moving the blocking block in the same direction if possible
                chain_pos = (next_pos[0] + dx, next_pos[1] + dy)
                
                # Check if chain position is valid
                if (0 <= chain_pos[0] < self.grid_size[0] and 
                    0 <= chain_pos[1] < self.grid_size[1] and
                    chain_pos not in state_set and
                    chain_pos not in self.obstacles):
                    
                    # Create new state by moving both blocks in the same direction
                    new_state_set = state_set.copy()
                    new_state_set.remove(pos)
                    new_state_set.remove(next_pos)
                    new_state_set.add(next_pos)
                    new_state_set.add(chain_pos)
                    
                    # Check if new state is connected and has the correct number of blocks
                    if self.is_connected(new_state_set) and len(new_state_set) == len(state_set):
                        valid_moves.append(frozenset(new_state_set))
                
                # Also try all other directions for the blocking block
                for chain_dx, chain_dy in self.directions:
                    if (chain_dx, chain_dy) == (dx, dy):
                        continue  # Skip the direction we just tried
                    
                    chain_pos = (next_pos[0] + chain_dx, next_pos[1] + chain_dy)
                    
                    # Skip if out of bounds, occupied, is an obstacle, or original position
                    if not (0 <= chain_pos[0] < self.grid_size[0] and 
                            0 <= chain_pos[1] < self.grid_size[1]):
                        continue
                    if chain_pos in state_set or chain_pos == pos or chain_pos in self.obstacles:
                        continue
                    
                    # Create new state by moving both blocks
                    new_state_set = state_set.copy()
                    new_state_set.remove(pos)
                    new_state_set.remove(next_pos)
                    new_state_set.add(next_pos)
                    new_state_set.add(chain_pos)
                    
                    # Check if new state is connected and has the correct number of blocks
                    if self.is_connected(new_state_set) and len(new_state_set) == len(state_set):
                        valid_moves.append(frozenset(new_state_set))
            
            # If next position is unoccupied, try direct move
            else:
                new_state_set = state_set.copy()
                new_state_set.remove(pos)
                new_state_set.add(next_pos)
                
                # Check if new state is connected and has the correct number of blocks
                if self.is_connected(new_state_set) and len(new_state_set) == len(state_set):
                    valid_moves.append(frozenset(new_state_set))
        
        return valid_moves
    
    def get_sliding_chain_moves(self, state):
        """
        Generate sliding chain moves where multiple blocks move in sequence
        to navigate tight spaces
        """
        state_set = set(state)
        valid_moves = []
        
        # NEW: Identify blocks that have reached their goal positions
        blocks_at_goal = state_set.intersection(self.goal_state)
        
        # For each block, try to initiate a sliding chain
        for pos in state_set:
            # Skip if it's at a goal position
            if pos in blocks_at_goal:
                continue
                
            # Skip if it's a critical articulation point
            articulation_points = self.get_articulation_points(state_set)
            if pos in articulation_points and len(articulation_points) <= 20:
                continue
                
            # Try sliding in each direction
            for dx, dy in self.directions:
                # Only consider diagonal moves for sliding chains
                if dx != 0 and dy != 0:
                    # Define the sliding path (up to 3 steps)
                    path = []
                    current_pos = pos
                    for _ in range(20):  # Maximum chain length
                        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                        # Stop if out of bounds or is an obstacle
                        if not (0 <= next_pos[0] < self.grid_size[0] and 
                                0 <= next_pos[1] < self.grid_size[1]):
                            break
                        if next_pos in self.obstacles:
                            break
                        path.append(next_pos)
                        current_pos = next_pos
                    
                    # Try sliding the block along the path
                    for i, target_pos in enumerate(path):
                        # Skip if target is occupied
                        if target_pos in state_set:
                            continue
                            
                        # Create new state by moving the block
                        new_state_set = state_set.copy()
                        new_state_set.remove(pos)
                        new_state_set.add(target_pos)
                        
                        # Check if new state is connected and has the correct number of blocks
                        if self.is_connected(new_state_set) and len(new_state_set) == len(state_set):
                            valid_moves.append(frozenset(new_state_set))
                        
                        # No need to continue if we can't reach this position
                        break
        
        return valid_moves
    
    def get_all_valid_moves(self, state):
        """
        Combine all move generation methods to maximize options
        """
        # Start with basic morphing moves
        basic_moves = self.get_valid_morphing_moves(state)
        
        # Add chain moves
        chain_moves = self.get_smart_chain_moves(state)
        
        # Add sliding chain moves
        sliding_moves = self.get_sliding_chain_moves(state)
        
        # Combine all moves (frozensets automatically handle duplicates)
        all_moves = list(set(basic_moves + chain_moves + sliding_moves))
        
        # NEW: Verify all moves have the correct number of blocks
        valid_moves = []
        for move in all_moves:
            if len(move) == len(state):
                valid_moves.append(move)
            else:
                print(f"WARNING: Invalid move with {len(move)} blocks instead of {len(state)}")
        
        return valid_moves
    
    def block_heuristic(self, state):
        """
        Heuristic for block movement phase:
        Now accounts for obstacles between current position and goal
        """
        if not state:
            return float('inf')
            
        current_centroid = self.calculate_centroid(state)
        goal_centroid_int = (int(self.goal_centroid[0]), int(self.goal_centroid[1]))
        
        # If no obstacles, use simple Manhattan distance
        if not self.obstacles:
            return abs(current_centroid[0] - self.goal_centroid[0]) + abs(current_centroid[1] - self.goal_centroid[1])
        
        # With obstacles, calculate path distance to goal centroid
        # Round centroid to nearest grid cell for distance calculation
        current_centroid_int = (int(round(current_centroid[0])), int(round(current_centroid[1])))
        
        # Ensure centroid is within bounds
        current_centroid_int = (
            max(0, min(current_centroid_int[0], self.grid_size[0]-1)),
            max(0, min(current_centroid_int[1], self.grid_size[1]-1))
        )
        goal_centroid_int = (
            max(0, min(goal_centroid_int[0], self.grid_size[0]-1)),
            max(0, min(goal_centroid_int[1], self.grid_size[1]-1))
        )
        
        # Get obstacle-aware distance
        return self.obstacle_aware_distance(current_centroid_int, goal_centroid_int)
    
    def improved_morphing_heuristic(self, state):
        """
        Improved heuristic for morphing phase:
        Uses bipartite matching to find optimal assignment of blocks to goal positions,
        now accounts for obstacles
        """
        if not state:
            return float('inf')
            
        state_list = list(state)
        goal_list = list(self.goal_state)
        
        # Early exit if states have different sizes
        if len(state_list) != len(goal_list):
            return float('inf')
        
        # NEW: Reward blocks that are already in goal positions
        blocks_at_goal = set(state).intersection(self.goal_state)
        goal_bonus = -len(blocks_at_goal) * 10  # Large negative value (bonus) for blocks in place
        
        # Build distance matrix with obstacle-aware distances
        distances = []
        for pos in state_list:
            # NEW: If block is already at a goal position, give it maximum preference
            # to stay where it is by assigning 0 distance to its current position
            # and high distance to all other positions
            if pos in self.goal_state:
                row = [0 if goal_pos == pos else 1000 for goal_pos in goal_list]
            else:
                row = []
                for goal_pos in goal_list:
                    # Use obstacle-aware distance calculation
                    if self.obstacles:
                        dist = self.obstacle_aware_distance(pos, goal_pos)
                    else:
                        # Use faster Manhattan distance if no obstacles
                        dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                        
                    # NEW: If this goal position is already filled, make it less attractive
                    if goal_pos in blocks_at_goal and goal_pos != pos:
                        dist += 100  # Discourage moving to goals that are already filled
                        
                    row.append(dist)
            distances.append(row)
        
        # Use greedy assignment algorithm
        total_distance = 0
        assigned_cols = set()
        
        # Sort rows by minimum distance
        row_indices = list(range(len(state_list)))
        row_indices.sort(key=lambda i: min(distances[i]))
        
        for i in row_indices:
            # Find closest unassigned goal position
            min_dist = float('inf')
            best_j = -1
            
            for j in range(len(goal_list)):
                if j not in assigned_cols and distances[i][j] < min_dist:
                    min_dist = distances[i][j]
                    best_j = j
            
            if best_j != -1:
                assigned_cols.add(best_j)
                total_distance += min_dist
                
                # If a path is impossible (infinite distance), heavily penalize
                if min_dist == float('inf'):
                    return float('inf')
            else:
                # No assignment possible
                return float('inf')
        
        # Return total distance plus goal bonus
        return total_distance + goal_bonus
    
    def block_movement_phase(self, time_limit=15):
        """
        Phase 1: Move the entire block toward the goal centroid
        Returns the path of states to get near the goal area
        Modified to stop 1 grid cell before reaching the goal centroid
        """
        print("Starting Block Movement Phase...")
        start_time = time.time()

        # For disconnected goals, use the modified approach
        if self.is_goal_disconnected:
            return self.disconnected_block_movement_phase(time_limit)

        # Initialize A* search
        open_set = [(self.block_heuristic(self.start_state), 0, self.start_state)]
        closed_set = set()

        # Track path and g-scores
        g_score = {self.start_state: 0}
        came_from = {self.start_state: None}

        # Modified: We want to stop 1 grid cell before reaching the centroid
        # Instead of using a small threshold, we'll check if distance is between 1.0 and 2.0
        # This ensures we're approximately 1 grid cell away from the goal centroid
        min_distance = 1.0
        max_distance = 1.0

        while open_set and time.time() - start_time < time_limit:
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
    
            # Skip if already processed
            if current in closed_set:
                continue
        
            # Check if we're at the desired distance from the goal centroid
            current_centroid = self.calculate_centroid(current)
            centroid_distance = (abs(current_centroid[0] - self.goal_centroid[0]) + 
                            abs(current_centroid[1] - self.goal_centroid[1]))
                        
            if min_distance <= centroid_distance <= max_distance:
                print(f"Block stopped 1 grid cell before goal centroid. Distance: {centroid_distance}")
                return self.reconstruct_path(came_from, current)
        
            closed_set.add(current)
    
            # Process neighbor states (block moves)
            for neighbor in self.get_valid_block_moves(current):
                if neighbor in closed_set:
                    continue
            
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
        
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                
                    # Modified: Adjust heuristic to prefer states that are close to but not at the centroid
                    neighbor_centroid = self.calculate_centroid(neighbor)
                    neighbor_distance = (abs(neighbor_centroid[0] - self.goal_centroid[0]) + 
                                    abs(neighbor_centroid[1] - self.goal_centroid[1]))
                
                    # Penalize distances that are too small (< 1.0)
                    distance_penalty = 0
                    if neighbor_distance < min_distance:
                        distance_penalty = 10 * (min_distance - neighbor_distance)
                
                    adjusted_heuristic = self.block_heuristic(neighbor) + distance_penalty
                    f_score = tentative_g + adjusted_heuristic
            
                    # Add to open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # If we exit the loop, either no path was found or time limit reached
        if time.time() - start_time >= time_limit:
            print("Block movement phase timed out!")
    
        # Return the best state we found
        if came_from:
            # Find state with appropriate distance to centroid
            best_state = None
            best_distance_diff = float('inf')
        
            for state in came_from.keys():
                state_centroid = self.calculate_centroid(state)
                distance = (abs(state_centroid[0] - self.goal_centroid[0]) + 
                            abs(state_centroid[1] - self.goal_centroid[1]))
            
                # We want a state that's as close as possible to our target distance range
                if distance < min_distance:
                    distance_diff = min_distance - distance
                elif distance > max_distance:
                    distance_diff = distance - max_distance
                else:
                    # Distance is within our desired range
                    best_state = state
                    break
                
                if distance_diff < best_distance_diff:
                    best_distance_diff = distance_diff
                    best_state = state
        
            if best_state:
                best_centroid = self.calculate_centroid(best_state)
                best_distance = (abs(best_centroid[0] - self.goal_centroid[0]) + 
                                abs(best_centroid[1] - self.goal_centroid[1]))
                print(f"Best block position found with centroid distance: {best_distance}")
                return self.reconstruct_path(came_from, best_state)
    
        return [self.start_state]  # No movement possible
    
    def smarter_morphing_phase(self, start_state, time_limit=15):
        """
        Improved Phase 2: Morph the block into the goal shape while maintaining connectivity
        Uses beam search and intelligent move generation with support for simultaneous moves
        Now with adaptive beam width based on obstacle density
        """
        print(f"Starting Smarter Morphing Phase with {self.min_simultaneous_moves}-{self.max_simultaneous_moves} simultaneous moves...")
        start_time = time.time()
        
        # NEW: Identify blocks already at goal positions in the start state
        self.blocks_at_goal = set(start_state).intersection(self.goal_state)
        print(f"Starting morphing with {len(self.blocks_at_goal)} blocks already at goal positions")
        
        # Adapt beam width based on obstacle density
        adaptive_beam_width = self.beam_width
        if len(self.obstacles) > 0:
            # Increase beam width for environments with obstacles
            obstacle_density = len(self.obstacles) / (self.grid_size[0] * self.grid_size[1])
            adaptive_beam_width = int(self.beam_width * (1 + min(1.0, obstacle_density * 5)))
            print(f"Adjusted beam width to {adaptive_beam_width} based on obstacle density")
            
        # Initialize beam search
        open_set = [(self.improved_morphing_heuristic(start_state), 0, start_state)]
        closed_set = set()
        
        # Track path, g-scores, and best state
        g_score = {start_state: 0}
        came_from = {start_state: None}
        
        # Track best state seen so far
        best_state = start_state
        best_heuristic = self.improved_morphing_heuristic(start_state)
        
        # NEW: Track the maximum number of blocks at goal positions seen so far
        max_blocks_at_goal = len(self.blocks_at_goal)
        
        iterations = 0
        last_improvement_time = time.time()
        
        while open_set and time.time() - start_time < time_limit:
            iterations += 1
            
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            # Check if goal reached
            if current == self.goal_state:
                print(f"Goal reached after {iterations} iterations!")
                return self.reconstruct_path(came_from, current)
            
            # Check how many blocks are at goal positions in current state
            blocks_at_goal_current = len(set(current).intersection(self.goal_state))
            
            # NEW: Update max_blocks_at_goal if we found a better state
            if blocks_at_goal_current > max_blocks_at_goal:
                max_blocks_at_goal = blocks_at_goal_current
                print(f"New maximum blocks at goal: {max_blocks_at_goal}/{len(self.goal_state)}")
            
            # Check if this is the best state seen so far
            current_heuristic = self.improved_morphing_heuristic(current)
            if current_heuristic < best_heuristic or blocks_at_goal_current > len(self.blocks_at_goal):
                best_state = current
                best_heuristic = current_heuristic
                self.blocks_at_goal = set(current).intersection(self.goal_state)
                last_improvement_time = time.time()
                
                # Print progress occasionally
                if iterations % 500 == 0:
                    print(f"Progress: h={best_heuristic}, blocks at goal={len(self.blocks_at_goal)}/{len(self.goal_state)}, iterations={iterations}")
                    
                # If we're very close to the goal, increase search intensity
                if best_heuristic < 5 * len(self.goal_state):
                    adaptive_beam_width *= 2
            
            # Check for stagnation - more patient in obstacle-heavy environments
            stagnation_tolerance = time_limit * (0.3 + min(0.3, len(self.obstacles) / 100))
            if time.time() - last_improvement_time > stagnation_tolerance:
                print("Search stagnated, restarting...")
                # Clear the beam and start from the best state
                open_set = [(best_heuristic, g_score[best_state], best_state)]
                last_improvement_time = time.time()
            
            # Limit iterations to prevent infinite loops
            if iterations >= self.max_iterations:
                print(f"Reached max iterations ({self.max_iterations})")
                break
                
            closed_set.add(current)
            
            # Get all valid moves
            neighbors = self.get_all_valid_moves(current)
            
            # Process each neighbor
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Check if the move causes blocks to disappear
                    if len(neighbor) != len(current):
                        continue  # Skip this move
                        
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # NEW: Calculate blocks at goal in this neighbor
                    blocks_at_goal_neighbor = len(set(neighbor).intersection(self.goal_state))
                    
                    # NEW: Prioritize states with more blocks at goal positions by giving them better f-scores
                    goal_position_bonus = max(0, blocks_at_goal_neighbor - blocks_at_goal_current) * 10
                    
                    f_score = tentative_g + self.improved_morphing_heuristic(neighbor) - goal_position_bonus
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
            
            # Beam search pruning: keep only the best states
            if len(open_set) > adaptive_beam_width:
                open_set = heapq.nsmallest(adaptive_beam_width, open_set)
                heapq.heapify(open_set)
        
        # If we exit the loop, either no path was found or time limit reached
        if time.time() - start_time >= time_limit:
            print(f"Morphing phase timed out after {iterations} iterations!")
        
        # Return the best state found
        return self.reconstruct_path(came_from, best_state)
    
    def reconstruct_path(self, came_from, current):
        """
        Reconstruct the path from start to goal
        """
        path = []
        while current:
            path.append(list(current))
            current = came_from.get(current)
        
        path.reverse()
        return path
    
    def search(self, time_limit=30):
        """
        Main search method combining block movement and smarter morphing
        Now with dynamic time allocation based on obstacles
        """
        # Build obstacle maze representation if not already done
        if self.obstacles and not self.obstacle_maze:
            self.build_obstacle_maze()
            
        # Dynamically allocate time based on obstacle density
        block_time_ratio = 0.3  # Default 30% for block movement
        
        # If there are obstacles, allocate more time for movement phase
        if len(self.obstacles) > 0:
            obstacle_density = len(self.obstacles) / (self.grid_size[0] * self.grid_size[1])
            # Allocate up to 50% for block movement in dense obstacle environments
            block_time_ratio = min(0.5, 0.3 + obstacle_density * 0.5)
            
        # For disconnected goals, adjust time allocation
        if self.is_goal_disconnected:
            return self.search_disconnected_goal(time_limit)
            
        block_time_limit = time_limit * block_time_ratio
        morphing_time_limit = time_limit * (1 - block_time_ratio)
        
        print(f"Time allocation: {block_time_ratio:.1%} block movement, {1-block_time_ratio:.1%} morphing")
        
        # Phase 1: Block Movement
        block_path = self.block_movement_phase(block_time_limit)
        
        if not block_path:
            print("Block movement phase failed!")
            return None
        
        # Get the final state from block movement phase
        block_final_state = frozenset(block_path[-1])
        
        # Phase 2: Smarter Morphing
        morphing_path = self.smarter_morphing_phase(block_final_state, morphing_time_limit)
        
        if not morphing_path:
            print("Morphing phase failed!")
            return block_path
        
        # Combine paths (remove duplicate state at transition)
        combined_path = block_path[:-1] + morphing_path
        
        # NEW: Verify block count is consistent throughout the path
        expected_count = len(self.start_state)
        for i, state in enumerate(combined_path):
            if len(state) != expected_count:
                print(f"WARNING: State {i} has {len(state)} blocks instead of {expected_count}")
                # Fix the state by using the previous valid state
                if i > 0:
                    combined_path[i] = combined_path[i-1]
        
        return combined_path
    
    def visualize_path(self, path, interval=0.5):
        """
        Visualize the path as an animation
        """
        if not path:
            print("No path to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.ion()  # Turn on interactive mode
    
        # Get bounds for plotting
        min_x, max_x = 0, self.grid_size[0] - 1
        min_y, max_y = 0, self.grid_size[1] - 1
    
        # Show initial state
        ax.clear()
        ax.set_xlim(min_x - 0.5, max_x + 0.5)
        ax.set_ylim(min_y - 0.5, max_y + 0.5)
        ax.grid(True)
    
        # NEW: Track and display blocks at goal positions differently
        # Draw goal positions (as outlines)
        goal_rects = []
        for pos in self.goal_positions:
            rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)
            goal_rects.append(rect)
    
        # Draw current positions
        current_positions = path[0]
        
        # NEW: Determine which blocks are at goal positions
        blocks_at_goal = [pos for pos in current_positions if (pos[0], pos[1]) in self.goal_state]
        blocks_not_at_goal = [pos for pos in current_positions if (pos[0], pos[1]) not in self.goal_state]
        
        # Draw blocks at goal positions (green filled squares)
        goal_block_rects = []
        for pos in blocks_at_goal:
            rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='green', alpha=0.7)
            ax.add_patch(rect)
            goal_block_rects.append(rect)
            
        # Draw other blocks (blue squares)
        non_goal_rects = []
        for pos in blocks_not_at_goal:
            rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='blue', alpha=0.7)
            ax.add_patch(rect)
            non_goal_rects.append(rect)
        
        ax.set_title(f"Step 0/{len(path)-1} - {len(blocks_at_goal)} blocks at goal")
        plt.draw()
        plt.pause(interval)
    
        # Animate the path
        for i in range(1, len(path)):
            # NEW: Verify block count is consistent
            if len(path[i]) != len(path[0]):
                print(f"Warning: State {i} has {len(path[i])} blocks instead of {len(path[0])}")
                # If block count is inconsistent, skip this frame
                continue
                
            # Update positions
            new_positions = path[i]
        
            # Clear previous blocks
            for rect in goal_block_rects + non_goal_rects:
                rect.remove()
            
            # NEW: Determine which blocks are at goal positions
            blocks_at_goal = [pos for pos in new_positions if (pos[0], pos[1]) in self.goal_state]
            blocks_not_at_goal = [pos for pos in new_positions if (pos[0], pos[1]) not in self.goal_state]
            
            # Draw blocks at goal positions (green filled squares)
            goal_block_rects = []
            for pos in blocks_at_goal:
                rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='green', alpha=0.7)
                ax.add_patch(rect)
                goal_block_rects.append(rect)
                
            # Draw other blocks (blue squares)
            non_goal_rects = []
            for pos in blocks_not_at_goal:
                rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='blue', alpha=0.7)
                ax.add_patch(rect)
                non_goal_rects.append(rect)
            
            ax.set_title(f"Step {i}/{len(path)-1} - {len(blocks_at_goal)} blocks at goal")
            plt.draw()
            plt.pause(interval)
    
        plt.ioff()  # Turn off interactive mode
        plt.show(block=True)
    
    def build_obstacle_maze(self):
        """Create a grid representation with obstacles for pathfinding"""
        self.obstacle_maze = [[0 for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        for x, y in self.obstacles:
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                self.obstacle_maze[x][y] = 1  # Mark obstacle cells
        
        # Clear the distance map cache when obstacles change
        self.distance_map_cache = {}
        
    def calculate_distance_map(self, target):
        """
        Calculate distance map from all cells to the target,
        accounting for obstacles (using BFS for accurate distances)
        """
        # Check if we've already computed this map
        if target in self.distance_map_cache:
            return self.distance_map_cache[target]
            
        # Initialize distance map with infinity
        dist_map = [[float('inf') for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]
        
        # BFS to calculate distances
        queue = deque([(target, 0)])  # (position, distance)
        visited = {target}
        
        while queue:
            (x, y), dist = queue.popleft()
            dist_map[x][y] = dist
            
            # Check all adjacent cells based on topology
            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                
                # Skip if out of bounds or is an obstacle or already visited
                if not (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]):
                    continue
                if (nx, ny) in self.obstacles or (nx, ny) in visited:
                    continue
                    
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
        
        # Cache the result
        self.distance_map_cache[target] = dist_map
        return dist_map
        
    def obstacle_aware_distance(self, pos, target):
        """
        Calculate the distance between a position and a target,
        accounting for obstacles
        """
        # If no obstacles, use Manhattan distance for speed
        if not self.obstacles:
            return abs(pos[0] - target[0]) + abs(pos[1] - target[1])
            
        # Get or calculate distance map for this target
        dist_map = self.calculate_distance_map(target)
        
        # Return the distance from the map
        return dist_map[pos[0]][pos[1]]
        
    # New methods for handling disconnected goal states
    
    def find_disconnected_components(self, positions):
        """
        Find all disconnected components in a set of positions using BFS
        Returns a list of sets, where each set contains positions in one component
        """
        if not positions:
            return []
            
        positions_set = set(positions)
        components = []
        
        while positions_set:
            # Start a new component
            component = set()
            start = next(iter(positions_set))
            
            # BFS to find all connected positions
            queue = deque([start])
            component.add(start)
            positions_set.remove(start)
            
            while queue:
                current = queue.popleft()
                
                # Check all adjacent positions
                for dx, dy in self.directions:
                    neighbor = (current[0] + dx, current[1] + dy)
                    if neighbor in positions_set:
                        component.add(neighbor)
                        positions_set.remove(neighbor)
                        queue.append(neighbor)
            
            # Add the component to the list
            components.append(component)
        
        return components
    
    def disconnected_block_movement_phase(self, time_limit=15):
        """
        Modified Phase 1 for disconnected goal states:
        Moves the entire block toward a strategic position for splitting
        """
        print("Starting Disconnected Block Movement Phase...")
        start_time = time.time()
        
        # Find the closest goal component to the start state
        closest_component_idx = self.find_closest_component()
        closest_component = self.goal_components[closest_component_idx]
        closest_centroid = self.component_centroids[closest_component_idx]
        
        # Determine if vertical positioning is better based on y-axis centroids
        all_components_y = [centroid[1] for centroid in self.component_centroids]
        overall_y = self.goal_centroid[1]
        
        # Check if centroid of all shapes is closer to y level of the closest shape
        use_vertical_approach = abs(overall_y - closest_centroid[1]) < sum([abs(y - closest_centroid[1]) for y in all_components_y]) / len(all_components_y)
        
        if use_vertical_approach:
            print("Using vertical approach for block movement")
            # Target position is at the overall centroid with y-level of closest component
            target_centroid = (self.goal_centroid[0], closest_centroid[1])
        else:
            print("Using standard approach for block movement")
            # Target position is the overall centroid
            target_centroid = self.goal_centroid
            
        # Cache original goal centroid and temporarily replace with target
        original_centroid = self.goal_centroid
        self.goal_centroid = target_centroid
        
        # Use standard A* search but with the modified target
        open_set = [(self.block_heuristic(self.start_state), 0, self.start_state)]
        closed_set = set()
        g_score = {self.start_state: 0}
        came_from = {self.start_state: None}
        
        # We want to get close to the target position
        min_distance = 1.0
        max_distance = 2.0

        while open_set and time.time() - start_time < time_limit:
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
    
            # Skip if already processed
            if current in closed_set:
                continue
        
            # Check if we're at the desired distance from the goal centroid
            current_centroid = self.calculate_centroid(current)
            centroid_distance = (abs(current_centroid[0] - target_centroid[0]) + 
                            abs(current_centroid[1] - target_centroid[1]))
                        
            if min_distance <= centroid_distance <= max_distance:
                print(f"Block stopped at strategic position. Distance: {centroid_distance}")
                # Restore original goal centroid
                self.goal_centroid = original_centroid
                return self.reconstruct_path(came_from, current)
        
            closed_set.add(current)
    
            # Process neighbor states (block moves)
            for neighbor in self.get_valid_block_moves(current):
                if neighbor in closed_set:
                    continue
            
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
        
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                
                    # Adjust heuristic to prefer states close to target
                    neighbor_centroid = self.calculate_centroid(neighbor)
                    neighbor_distance = (abs(neighbor_centroid[0] - target_centroid[0]) + 
                                    abs(neighbor_centroid[1] - target_centroid[1]))
                
                    # Penalize distances that are too small
                    distance_penalty = 0
                    if neighbor_distance < min_distance:
                        distance_penalty = 10 * (min_distance - neighbor_distance)
                
                    # Calculate Manhattan distance to target
                    if self.obstacles:
                        neighbor_centroid_int = (int(round(neighbor_centroid[0])), int(round(neighbor_centroid[1])))
                        target_centroid_int = (int(round(target_centroid[0])), int(round(target_centroid[1])))
                        
                        # Ensure centroids are within bounds
                        neighbor_centroid_int = (
                            max(0, min(neighbor_centroid_int[0], self.grid_size[0]-1)),
                            max(0, min(neighbor_centroid_int[1], self.grid_size[1]-1))
                        )
                        target_centroid_int = (
                            max(0, min(target_centroid_int[0], self.grid_size[0]-1)),
                            max(0, min(target_centroid_int[1], self.grid_size[1]-1))
                        )
                        
                        adjusted_heuristic = self.obstacle_aware_distance(neighbor_centroid_int, target_centroid_int) + distance_penalty
                    else:
                        adjusted_heuristic = neighbor_distance + distance_penalty
                        
                    f_score = tentative_g + adjusted_heuristic
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # Restore original goal centroid
        self.goal_centroid = original_centroid
        
        # If we exit the loop, find the best available state
        if came_from:
            best_state = None
            best_distance_diff = float('inf')
        
            for state in came_from.keys():
                state_centroid = self.calculate_centroid(state)
                distance = (abs(state_centroid[0] - target_centroid[0]) + 
                            abs(state_centroid[1] - target_centroid[1]))
            
                if distance < min_distance:
                    distance_diff = min_distance - distance
                elif distance > max_distance:
                    distance_diff = distance - max_distance
                else:
                    best_state = state
                    break
                
                if distance_diff < best_distance_diff:
                    best_distance_diff = distance_diff
                    best_state = state
        
            if best_state:
                print(f"Best strategic position found for disconnected goal")
                return self.reconstruct_path(came_from, best_state)
    
        return [self.start_state]  # No movement possible
    
    def find_closest_component(self):
        """Find the index of the closest goal component to the start state"""
        start_centroid = self.calculate_centroid(self.start_state)
        min_distance = float('inf')
        closest_idx = 0
        
        for idx, centroid in enumerate(self.component_centroids):
            distance = abs(start_centroid[0] - centroid[0]) + abs(start_centroid[1] - centroid[1])
            if distance < min_distance:
                min_distance = distance
                closest_idx = idx
                
        return closest_idx
    
    def assign_blocks_to_components(self, state):
        """
        Returns a dictionary mapping each component index to a set of positions
        """
        assignments = {i: set() for i in range(len(self.goal_components))}
        state_positions = list(state)
        
        # Create a dictionary to track assigned positions
        assigned = set()
        
        # First, count how many blocks we need for each component
        component_sizes = [len(comp) for comp in self.goal_components]
        total_blocks_needed = sum(component_sizes)
        
        # Ensure we have enough blocks
        if len(state_positions) < total_blocks_needed:
            print(f"Warning: Not enough blocks ({len(state_positions)}) for goal state ({total_blocks_needed})")
            return None
            
        # Calculate distance from each block to each component centroid
        distances = []
        for pos in state_positions:
            pos_distances = []
            for idx, centroid in enumerate(self.component_centroids):
                if self.obstacles:
                    dist = self.obstacle_aware_distance(pos, (int(centroid[0]), int(centroid[1])))
                else:
                    dist = abs(pos[0] - centroid[0]) + abs(pos[1] - centroid[1])
                pos_distances.append((idx, dist))
            distances.append((pos, sorted(pos_distances, key=lambda x: x[1])))
            
        # Sort blocks by their distance to their closest component
        distances.sort(key=lambda x: x[1][0][1])
        
        # Assign blocks to components in order of increasing distance
        for pos, component_distances in distances:
            for component_idx, dist in component_distances:
                if len(assignments[component_idx]) < len(self.goal_components[component_idx]) and pos not in assigned:
                    assignments[component_idx].add(pos)
                    assigned.add(pos)
                    break
                    
        # Ensure all blocks are assigned
        unassigned = set(state_positions) - assigned
        if unassigned:
            # Assign remaining blocks to components that still need them
            for pos in unassigned:
                for component_idx in range(len(self.goal_components)):
                    if len(assignments[component_idx]) < len(self.goal_components[component_idx]):
                        assignments[component_idx].add(pos)
                        assigned.add(pos)
                        break
                        
        # Double-check that we've assigned the right number of blocks to each component
        for idx, component in enumerate(self.goal_components):
            if len(assignments[idx]) != len(component):
                print(f"Warning: Component {idx} has {len(assignments[idx])} blocks assigned but needs {len(component)}")
                
        return assignments
    
    def plan_disconnect_moves(self, state, assignments):
        """
        Plan a sequence of moves to disconnect the shape into separate components
        Returns a list of states representing the disconnection process
        """
        # Start with current state
        current_state = set(state)
        path = [frozenset(current_state)]
        
        # Group the assignments by component
        component_positions = [set(assignments[i]) for i in range(len(self.goal_components))]
        
        # Check if the state is already naturally separable
        is_separable = True
        for i in range(len(component_positions)):
            for j in range(i+1, len(component_positions)):
                # Check if there's a direct connection between components
                for pos_i in component_positions[i]:
                    for pos_j in component_positions[j]:
                        # Check if positions are adjacent
                        if any((pos_i[0] + dx, pos_i[1] + dy) == pos_j for dx, dy in self.directions):
                            is_separable = False
                            break
                    if not is_separable:
                        break
                if not is_separable:
                    break
            if not is_separable:
                break
                
        # If already separable, return current state
        if is_separable:
            print("State is already naturally separable into components")
            return path
            
        # Find minimal set of points that connect the components
        connection_points = set()
        for i in range(len(component_positions)):
            for j in range(i+1, len(component_positions)):
                # Find all connections between components i and j
                for pos_i in component_positions[i]:
                    for pos_j in component_positions[j]:
                        # Check if positions are adjacent
                        if any((pos_i[0] + dx, pos_i[1] + dy) == pos_j for dx, dy in self.directions):
                            # Add both positions to connection points
                            connection_points.add(pos_i)
                            connection_points.add(pos_j)
        
        # For theoretical disconnection, we don't need to actually move blocks
        # Just mark the connection points as if they're disconnected
        print(f"Identified {len(connection_points)} connection points for theoretical disconnection")
        
        # Return current state as the disconnection plan
        # The actual disconnection will happen during the morphing phase
        return path
    
    def component_morphing_heuristic(self, state, goal_component):
        """
        Heuristic for morphing a specific component
        """
        if not state:
            return float('inf')
            
        state_list = list(state)
        goal_list = list(goal_component)
        
        # Early exit if states have different sizes
        if len(state_list) != len(goal_list):
            return float('inf')
        
        # Build distance matrix with obstacle-aware distances
        distances = []
        for pos in state_list:
            row = []
            for goal_pos in goal_list:
                # Use obstacle-aware distance calculation
                if self.obstacles:
                    dist = self.obstacle_aware_distance(pos, goal_pos)
                else:
                    # Use faster Manhattan distance if no obstacles
                    dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                row.append(dist)
            distances.append(row)
        
        # Use greedy assignment algorithm
        total_distance = 0
        assigned_cols = set()
        
        # Sort rows by minimum distance
        row_indices = list(range(len(state_list)))
        row_indices.sort(key=lambda i: min(distances[i]))
        
        for i in row_indices:
            # Find closest unassigned goal position
            min_dist = float('inf')
            best_j = -1
            
            for j in range(len(goal_list)):
                if j not in assigned_cols and distances[i][j] < min_dist:
                    min_dist = distances[i][j]
                    best_j = j
            
            if best_j != -1:
                assigned_cols.add(best_j)
                total_distance += min_dist
                
                # If a path is impossible, heavily penalize
                if min_dist == float('inf'):
                    return float('inf')
            else:
                # No assignment possible
                return float('inf')
        
        # Add connectivity bonus: prefer states that have more blocks in goal positions
        goal_component_set = frozenset(goal_component)
        matching_positions = len(frozenset(state).intersection(goal_component_set))
        connectivity_bonus = -matching_positions * 0.5  # Negative to encourage more matches
        
        return total_distance + connectivity_bonus
    
    def component_morphing_phase(self, start_state, goal_component, time_limit=15):
        """
        Morph a specific component into its goal shape
        """
        start_time = time.time()
        
        # Adapt beam width based on obstacle density
        adaptive_beam_width = self.beam_width
        if len(self.obstacles) > 0:
            obstacle_density = len(self.obstacles) / (self.grid_size[0] * self.grid_size[1])
            adaptive_beam_width = int(self.beam_width * (1 + min(1.0, obstacle_density * 5)))
            
        # Initialize beam search
        open_set = [(self.component_morphing_heuristic(start_state, goal_component), 0, start_state)]
        closed_set = set()
        
        # Track path, g-scores, and best state
        g_score = {start_state: 0}
        came_from = {start_state: None}
        
        # Track best state seen so far
        best_state = start_state
        best_heuristic = self.component_morphing_heuristic(start_state, goal_component)
        
        iterations = 0
        last_improvement_time = time.time()
        goal_component_set = frozenset(goal_component)
        
        while open_set and time.time() - start_time < time_limit:
            iterations += 1
            
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            # Check if goal reached
            if current == goal_component_set:
                print(f"Component goal reached after {iterations} iterations!")
                return self.reconstruct_path(came_from, current)
            
            # Check if this is the best state seen so far
            current_heuristic = self.component_morphing_heuristic(current, goal_component)
            if current_heuristic < best_heuristic:
                best_state = current
                best_heuristic = current_heuristic
                last_improvement_time = time.time()
                
                # Print progress occasionally
                if iterations % 500 == 0:
                    print(f"Component progress: h={best_heuristic}, iterations={iterations}")
                
            # If we're very close to the goal, increase search intensity
                if best_heuristic < 5 * len(goal_component):
                    adaptive_beam_width *= 2
        
            # Check for stagnation
            stagnation_tolerance = time_limit * 0.3
            if time.time() - last_improvement_time > stagnation_tolerance:
                print("Component search stagnated, restarting...")
                # Clear the beam and start from the best state
                open_set = [(best_heuristic, g_score[best_state], best_state)]
                last_improvement_time = time.time()
        
            # Limit iterations to prevent infinite loops
            if iterations >= self.max_iterations:
                print(f"Reached max iterations ({self.max_iterations}) for component morphing")
                break
            
            closed_set.add(current)
        
            # Get valid moves for this component
            neighbors = self.get_all_valid_moves(current)
        
            # Process each neighbor
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
            
                # Skip if neighbor has wrong size
                if len(neighbor) != len(goal_component):
                    continue
            
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
            
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.component_morphing_heuristic(neighbor, goal_component)
                
                    # Add to open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
            # Beam search pruning: keep only the best states
            if len(open_set) > adaptive_beam_width:
                open_set = heapq.nsmallest(adaptive_beam_width, open_set)
                heapq.heapify(open_set)
    
        # If we exit the loop, return the best state found
        print(f"Component morphing timed out after {iterations} iterations!")
        return self.reconstruct_path(came_from, best_state)
    
    def search_disconnected_goal(self, time_limit=30):
        """
        Search method for disconnected goal states:
        1. Move blocks to strategic position
        2. Assign blocks to components
        3. Morph each component in parallel
        """
        print(f"Starting search for disconnected goal with {len(self.goal_components)} components")
        start_time = time.time()
    
        # Allocate time for different phases
        move_time_ratio = 0.2  # 20% for block movement
        disconnect_time_ratio = 0.1  # 10% for disconnection planning
        morphing_time_ratio = 0.7  # 70% for morphing
    
        # Adjust ratios if obstacles are present
        if len(self.obstacles) > 0:
            obstacle_density = len(self.obstacles) / (self.grid_size[0] * self.grid_size[1])
            move_time_ratio = min(0.4, 0.2 + obstacle_density * 0.5)  # Up to 40% for movement
            morphing_time_ratio = 1.0 - move_time_ratio - disconnect_time_ratio
    
        move_time_limit = time_limit * move_time_ratio
        disconnect_time_limit = time_limit * disconnect_time_ratio
        morphing_time_limit = time_limit * morphing_time_ratio
    
        print(f"Time allocation: {move_time_ratio:.1%} block movement, "
                f"{disconnect_time_ratio:.1%} disconnection planning, "
                f"{morphing_time_ratio:.1%} morphing")
    
        # Phase 1: Strategic Block Movement
        block_path = self.disconnected_block_movement_phase(move_time_limit)
    
        if not block_path:
            print("Block movement phase failed!")
            return None
    
        # Get the final state from block movement phase
        block_final_state = frozenset(block_path[-1])
    
        # Phase 2: Assign blocks to components and plan disconnection
        assignments_start_time = time.time()
        assignments = self.assign_blocks_to_components(block_final_state)
    
        if assignments is None:
            print("Failed to assign blocks to components!")
            return block_path
    
        # Theoretical disconnection planning
        disconnect_path = self.plan_disconnect_moves(block_final_state, assignments)
        disconnect_time_used = time.time() - assignments_start_time
    
        # Add remaining disconnect time to morphing time
        if disconnect_time_used < disconnect_time_limit:
            morphing_time_limit += (disconnect_time_limit - disconnect_time_used)
    
        # Phase 3: Parallel Morphing of Components
        print(f"Starting parallel morphing of {len(self.goal_components)} components")
    
        # Prepare component states and goals
        component_start_states = []
        for i in range(len(self.goal_components)):
            component_start_states.append(frozenset(assignments[i]))
    
        # Time allocation for each component based on its size
        total_blocks = sum(len(comp) for comp in self.goal_components)
        component_time_limits = [
            morphing_time_limit * len(comp) / total_blocks
            for comp in self.goal_components
        ]
    
        # Use thread pool for parallel morphing
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.goal_components)) as executor:
            # Submit tasks for each component
            futures = []
            for i in range(len(self.goal_components)):
                futures.append(
                    executor.submit(
                        self.component_morphing_phase,
                        component_start_states[i],
                        self.goal_components[i],
                        component_time_limits[i]
                    )
                )
        
            # Collect results as they complete
            component_paths = []
            for future in concurrent.futures.as_completed(futures):
                component_paths.append(future.result())
    
        # Combine the paths from all components
        combined_path = block_path[:-1]  # Remove the last state from block path
    
        # Add the component paths after the block movement
        # For visualization, we'll alternate steps from each component
        # to show them moving simultaneously
        max_component_path_length = max(len(path) for path in component_paths)
    
        for step in range(max_component_path_length):
            combined_state = set()
            for i, path in enumerate(component_paths):
                # If this component's path is long enough, add its state at this step
                if step < len(path):
                    combined_state.update(path[step])
        
            # Add the combined state to the path if not empty
            if combined_state:
                combined_path.append(list(combined_state))
    
        return combined_path
    
    def get_disconnected_valid_moves(self, state, goal_components):
        """
        Generate valid moves for disconnected components
        Allows moves that maintain connectivity within each component
        But doesn't require connectivity between components
        """
        # Find the components in the current state
        current_components = self.find_disconnected_components(state)
    
        # If we have fewer components than needed, can't generate valid disconnected moves
        if len(current_components) < len(goal_components):
            return self.get_all_valid_moves(state)
    
        # Generate moves for each component separately
        all_moves = []
    
        for component in current_components:
            component_moves = []
        
            # Get basic morphing moves for this component
            component_state = frozenset(component)
            basic_moves = self.get_valid_morphing_moves(component_state)
            chain_moves = self.get_smart_chain_moves(component_state)
            sliding_moves = self.get_sliding_chain_moves(component_state)
        
            # Combine all move types
            component_moves.extend(basic_moves)
            component_moves.extend(chain_moves)
            component_moves.extend(sliding_moves)
        
            # For each component move, create a new overall state
            for move in component_moves:
                # Create new overall state with this component's move
                new_state = (state - component_state) | move
                all_moves.append(frozenset(new_state))
    
        return all_moves