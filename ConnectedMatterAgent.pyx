import heapq
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Cython imports
cimport numpy as np
from libc.math cimport abs as c_abs
from libc.stdlib cimport malloc, free

# Helper function for DFS (outside the class)
cdef void _dfs(tuple u, list time, set state_set, set articulation_points, 
              set visited, dict discovery, dict low, dict parent, list directions):
    cdef int children = 0
    cdef tuple v
    cdef int dx, dy
    
    visited.add(u)
    discovery[u] = low[u] = time[0]
    time[0] += 1
    
    # Visit all neighbors
    for dx, dy in directions:
        v = (u[0] + dx, u[1] + dy)
        if v in state_set:
            if v not in visited:
                children += 1
                parent[v] = u
                _dfs(v, time, state_set, articulation_points, visited, discovery, low, parent, directions)
                
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

# Helper function to find min value in a row (replaces lambda)
cdef int min_value_in_row(list distances, int row_index):
    cdef int min_val = distances[row_index][0]
    cdef int j
    for j in range(1, len(distances[row_index])):
        if distances[row_index][j] < min_val:
            min_val = distances[row_index][j]
    return min_val

# Helper for sorting indices by row minimums
cdef list sort_indices_by_row_min(list distances, int length):
    cdef list result = list(range(length))
    cdef int i, j, temp
    
    # Simple bubble sort implementation
    for i in range(length):
        for j in range(0, length-i-1):
            if min_value_in_row(distances, result[j]) > min_value_in_row(distances, result[j+1]):
                temp = result[j]
                result[j] = result[j+1]
                result[j+1] = temp
                
    return result

# Helper function to generate move combinations
cdef list _generate_move_combinations(list single_moves, int k, int start_idx=0):
    cdef list result = []
    cdef int i
    cdef object move
    cdef list combo
    
    if k == 1:
        return [[move] for move in single_moves[start_idx:]]
    
    for i in range(start_idx, len(single_moves) - k + 1):
        move = single_moves[i]
        for combo in _generate_move_combinations(single_moves, k-1, i+1):
            result.append([move] + combo)
    
    return result

cdef class ConnectedMatterAgent:
    # Class attributes with type declarations
    cdef tuple grid_size
    cdef list start_positions, goal_positions
    cdef str topology
    cdef int max_simultaneous_moves, min_simultaneous_moves
    cdef list directions
    cdef object start_state, goal_state  # frozenset
    cdef tuple goal_centroid
    cdef dict valid_moves_cache, articulation_points_cache, connectivity_check_cache
    cdef int beam_width, max_iterations

    def __init__(self, tuple grid_size, list start_positions, list goal_positions, 
                str topology="moore", int max_simultaneous_moves=1, int min_simultaneous_moves=1):
        self.grid_size = grid_size
        self.start_positions = list(start_positions)
        self.goal_positions = list(goal_positions)
        self.topology = topology
        self.max_simultaneous_moves = max_simultaneous_moves
        self.min_simultaneous_moves = min(min_simultaneous_moves, max_simultaneous_moves)  # Ensure min <= max
        
        # Set moves based on topology
        if self.topology == "moore":
            self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:  # Von Neumann
            self.directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
            
        # Initialize the start and goal states
        self.start_state = frozenset((x, y) for x, y in start_positions)
        self.goal_state = frozenset((x, y) for x, y in goal_positions)
        
        # Calculate the centroid of the goal positions for block movement phase
        self.goal_centroid = self.calculate_centroid(self.goal_positions)
        
        # Cache for valid moves to avoid recomputation
        self.valid_moves_cache = {}
        
        # For optimizing the search
        self.articulation_points_cache = {}
        self.connectivity_check_cache = {}
        
        # Enhanced parameters for improved search
        self.beam_width = 500  # Increased beam width for better exploration
        self.max_iterations = 10000  # Limit iterations to prevent infinite loops
        
    cpdef tuple calculate_centroid(self, list positions):
        """Calculate the centroid (average position) of a set of positions"""
        if not positions:
            return (0, 0)
        
        cdef double x_sum = 0
        cdef double y_sum = 0
        cdef int length = len(positions)
        cdef int i
        
        for i in range(length):
            x_sum += positions[i][0]
            y_sum += positions[i][1]
            
        return (x_sum / length, y_sum / length)
    
    cpdef bint is_connected(self, list positions):
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
        
        cdef tuple current, neighbor
        cdef int dx, dy
        
        while queue:
            current = queue.popleft()
            
            # Check all adjacent positions
            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in positions_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # All positions should be visited if connected
        cdef bint is_connected_result = len(visited) == len(positions_set)
        
        # Cache the result
        self.connectivity_check_cache[positions_hash] = is_connected_result
        return is_connected_result
    
    cpdef set get_articulation_points(self, set state_set):
        """
        Find articulation points (critical points that if removed would disconnect the structure)
        Uses a modified DFS algorithm
        """
        cdef int state_hash = hash(frozenset(state_set))
        if state_hash in self.articulation_points_cache:
            return self.articulation_points_cache[state_hash]
            
        if len(state_set) <= 2:  # All points are critical in structures of size 1-2
            self.articulation_points_cache[state_hash] = set(state_set)
            return set(state_set)
            
        cdef set articulation_points = set()
        cdef set visited = set()
        cdef dict discovery = {}
        cdef dict low = {}
        cdef dict parent = {}
        cdef list time = [0]  # Using list to allow modification inside nested function
        cdef tuple point
        
        # Call DFS for all vertices (using the external helper function)
        for point in state_set:
            if point not in visited:
                _dfs(point, time, state_set, articulation_points, visited, discovery, low, parent, self.directions)
                
        self.articulation_points_cache[state_hash] = articulation_points
        return articulation_points
    
    cpdef list get_valid_block_moves(self, object state):
        """
        Generate valid moves for the entire block of elements
        A valid block move shifts all elements in the same direction while maintaining connectivity
        """
        cdef list valid_moves = []
        cdef list state_list = list(state)
        cdef int dx, dy
        cdef list new_positions
        cdef bint all_valid
        cdef object new_state
        
        # Try moving the entire block in each direction
        for dx, dy in self.directions:
            # Calculate new positions after moving
            new_positions = [(pos[0] + dx, pos[1] + dy) for pos in state_list]
            
            # Check if all new positions are valid (within bounds and not occupied)
            all_valid = True
            for pos in new_positions:
                if not (0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]):
                    all_valid = False
                    break
            
            # Only consider moves that keep all positions within bounds
            if all_valid:
                new_state = frozenset(new_positions)
                valid_moves.append(new_state)
        
        return valid_moves
    
    def get_valid_morphing_moves(self, state):
    # Convert state to a list if it's a frozenset or any other iterable
        cdef list occupied_positions = list(state)
        cdef set occupied_set = set(occupied_positions)
        cdef list potential_moves = []
        cdef list valid_moves = []
        cdef tuple pos, neighbor, new_pos
        cdef int x, y, nx, ny
        cdef list neighbors_list
        cdef bint is_valid
    
    # Find all boundary positions - these are where morphing can occur
        cdef list boundary_positions = []
        for pos in occupied_positions:
            x, y = pos
            neighbors_count = 0
        
            for dx, dy in self.neighbor_transforms:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    if neighbor in occupied_set:
                        neighbors_count += 1
        
            if neighbors_count < len(self.neighbor_transforms):
                boundary_positions.append(pos)
    
    # For each boundary position, find potential new positions
        for pos in boundary_positions:
            x, y = pos
            neighbors_list = []
        
            for dx, dy in self.neighbor_transforms:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
            
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height and neighbor not in occupied_set:
                # Check if removing pos and adding neighbor maintains connectivity
                    new_occupied = occupied_positions.copy()
                    new_occupied.remove(pos)
                    new_occupied.append(neighbor)
                
                    if self.is_connected(new_occupied):
                    # Further check to ensure that pos is not an articulation point
                    # or that removing it doesn't break connectivity
                        articulation_points = self.find_articulation_points(occupied_positions)
                    
                    # Handle articulation points that might be strings
                        articulation_points_fixed = []
                        for point in articulation_points:
                            if isinstance(point, str):
                            # Parse the string into a tuple - assuming format like "(x,y)"
                                coords = point.strip('()').split(',')
                                point_tuple = (int(coords[0]), int(coords[1]))
                                articulation_points_fixed.append(point_tuple)
                            else:
                                articulation_points_fixed.append(point)
                    
                        if pos not in articulation_points_fixed:
                            potential_moves.append((pos, neighbor))
    
    # Filter potential moves to ensure connectivity
        for pos, new_pos in potential_moves:
            # Create a new state by moving from pos to new_pos
            new_state = occupied_positions.copy()
            new_state.remove(pos)
            new_state.append(new_pos)
        
            if self.is_valid_state(new_state):
                valid_moves.append((pos, new_pos))
    
        return valid_moves
    
    cpdef bint _is_valid_move_combination(self, list moves, set state_set):
        """Check if a combination of moves is valid (no conflicts)"""
        # Extract source and target positions
        cdef set sources = set()
        cdef set targets = set()
        cdef tuple src, tgt
        
        for src, tgt in moves:
            # Check for overlapping sources or targets
            if src in sources or tgt in targets:
                return False
            sources.add(src)
            targets.add(tgt)
            
            # Check that no target is also a source for another move
            if tgt in sources or src in targets:
                return False
        
        return True
    
    cpdef set _apply_moves(self, set state_set, list moves):
        """Apply a list of moves to the state"""
        cdef set new_state = state_set.copy()
        cdef tuple src, tgt
        
        for src, tgt in moves:
            new_state.remove(src)
            new_state.add(tgt)
        
        return new_state
    
    cpdef list get_smart_chain_moves(self, object state):
        """
        Generate chain moves where one block moves into the space of another
        while that block moves elsewhere, maintaining connectivity
        """
        cdef set state_set = set(state)
        cdef list valid_moves = []
        cdef tuple pos, closest_goal, next_pos, chain_pos
        cdef double min_dist, dist
        cdef int dx, dy, chain_dx, chain_dy
        cdef set new_state_set
        
        # For each block, try to move it toward a goal position
        for pos in state_set:
            # Find closest goal position
            min_dist = float('inf')
            closest_goal = None
            
            for goal_pos in self.goal_state:
                if goal_pos not in state_set:  # Only consider unoccupied goals
                    dist = c_abs(pos[0] - goal_pos[0]) + c_abs(pos[1] - goal_pos[1])
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
            
            # Skip if out of bounds
            if not (0 <= next_pos[0] < self.grid_size[0] and 
                    0 <= next_pos[1] < self.grid_size[1]):
                continue
            
            # If next position is occupied, try chain move
            if next_pos in state_set:
                # Try moving the blocking block elsewhere
                for chain_dx, chain_dy in self.directions:
                    chain_pos = (next_pos[0] + chain_dx, next_pos[1] + chain_dy)
                    
                    # Skip if out of bounds, occupied, or original position
                    if not (0 <= chain_pos[0] < self.grid_size[0] and 
                            0 <= chain_pos[1] < self.grid_size[1]):
                        continue
                    if chain_pos in state_set or chain_pos == pos:
                        continue
                    
                    # Create new state by moving both blocks
                    new_state_set = state_set.copy()
                    new_state_set.remove(pos)
                    new_state_set.remove(next_pos)
                    new_state_set.add(next_pos)
                    new_state_set.add(chain_pos)
                    
                    # Check if new state is connected
                    if self.is_connected(list(new_state_set)):
                        valid_moves.append(frozenset(new_state_set))
            
            # If next position is unoccupied, try direct move
            else:
                new_state_set = state_set.copy()
                new_state_set.remove(pos)
                new_state_set.add(next_pos)
                
                # Check if new state is connected
                if self.is_connected(list(new_state_set)):
                    valid_moves.append(frozenset(new_state_set))
        
        return valid_moves
    
    cpdef list get_sliding_chain_moves(self, object state):
        """
        Generate sliding chain moves where multiple blocks move in sequence
        to navigate tight spaces
        """
        cdef set state_set = set(state)
        cdef list valid_moves = []
        cdef tuple pos, current_pos, next_pos, target_pos
        cdef int dx, dy, i
        cdef set articulation_points, new_state_set
        cdef list path
        
        # For each block, try to initiate a sliding chain
        for pos in state_set:
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
                        # Stop if out of bounds
                        if not (0 <= next_pos[0] < self.grid_size[0] and 
                                0 <= next_pos[1] < self.grid_size[1]):
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
                        
                        # Check if new state is connected
                        if self.is_connected(list(new_state_set)):
                            valid_moves.append(frozenset(new_state_set))
                        
                        # No need to continue if we can't reach this position
                        break
        
        return valid_moves
    
    cpdef list get_all_valid_moves(self, object state):
        """
        Combine all move generation methods to maximize options
        """
        # Start with basic morphing moves
        cdef list basic_moves = self.get_valid_morphing_moves(state)
        
        # Add chain moves
        cdef list chain_moves = self.get_smart_chain_moves(state)
        
        # Add sliding chain moves
        cdef list sliding_moves = self.get_sliding_chain_moves(state)
        
        # Combine all moves (frozensets automatically handle duplicates)
        cdef list all_moves = list(set(basic_moves + chain_moves + sliding_moves))
        
        return all_moves
    
    cpdef double block_heuristic(self, object state):
        """
        Heuristic for block movement phase:
        Calculate Manhattan distance from current centroid to goal centroid
        """
        if not state:
            return float('inf')
            
        cdef tuple current_centroid = self.calculate_centroid(list(state))
        
        # Pure Manhattan distance between centroids without the +1 offset
        return c_abs(current_centroid[0] - self.goal_centroid[0]) + c_abs(current_centroid[1] - self.goal_centroid[1])
    
    cpdef double improved_morphing_heuristic(self, object state):
        """
        Improved heuristic for morphing phase:
        Uses bipartite matching to find optimal assignment of blocks to goal positions
        """
        if not state:
            return float('inf')
            
        cdef list state_list = list(state)
        cdef list goal_list = list(self.goal_state)
        
        # Early exit if states have different sizes
        if len(state_list) != len(goal_list):
            return float('inf')
        
        # Build distance matrix
        cdef list distances = []
        cdef list row
        cdef tuple pos, goal_pos
        cdef int dist
        
        for pos in state_list:
            row = []
            for goal_pos in goal_list:
                # Manhattan distance
                dist = c_abs(pos[0] - goal_pos[0]) + c_abs(pos[1] - goal_pos[1])
                row.append(dist)
            distances.append(row)
        
        # Use greedy assignment algorithm
        cdef double total_distance = 0
        cdef set assigned_cols = set()
        cdef int i, j, best_j, matching_positions
        cdef double min_dist, connectivity_bonus
        
        # Sort rows by minimum distance (using our helper function)
        cdef list row_indices = sort_indices_by_row_min(distances, len(state_list))
        
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
            else:
                # No assignment possible
                return float('inf')
        
        # Add connectivity bonus: prefer states that have more blocks in goal positions
        matching_positions = len(state.intersection(self.goal_state))
        connectivity_bonus = -matching_positions * 0.5  # Negative to encourage more matches
        
        return total_distance + connectivity_bonus
    
    cpdef list block_movement_phase(self, double time_limit=15):
        """
        Phase 1: Move the entire block toward the goal centroid
        Returns the path of states to get near the goal area
        Modified to stop 1 grid cell before reaching the goal centroid
        """
        print("Starting Block Movement Phase...")
        cdef double start_time = time.time()

        # Initialize A* search
        open_set = [(self.block_heuristic(self.start_state), 0, self.start_state)]
        cdef set closed_set = set()

        # Track path and g-scores
        cdef dict g_score = {self.start_state: 0}
        cdef dict came_from = {self.start_state: None}

        # Modified: We want to stop 1 grid cell before reaching the centroid
        # Instead of using a small threshold, we'll check if distance is between 1.0 and 2.0
        # This ensures we're approximately 1 grid cell away from the goal centroid
        cdef double min_distance = 1.0
        cdef double max_distance = 1.0
        cdef double f, g, centroid_distance, neighbor_distance, tentative_g, f_score
        cdef double distance_diff, best_distance_diff, distance_penalty, adjusted_heuristic
        cdef object current, neighbor
        cdef tuple current_centroid, neighbor_centroid, best_centroid
        cdef object best_state = None

        while open_set and time.time() - start_time < time_limit:
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
    
            # Skip if already processed
            if current in closed_set:
                continue
        
            # Check if we're at the desired distance from the goal centroid
            current_centroid = self.calculate_centroid(list(current))
            centroid_distance = (c_abs(current_centroid[0] - self.goal_centroid[0]) + 
                            c_abs(current_centroid[1] - self.goal_centroid[1]))
                        
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
                    neighbor_centroid = self.calculate_centroid(list(neighbor))
                    neighbor_distance = (c_abs(neighbor_centroid[0] - self.goal_centroid[0]) + 
                                   c_abs(neighbor_centroid[1] - self.goal_centroid[1]))
                
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
                state_centroid = self.calculate_centroid(list(state))
                distance = (c_abs(state_centroid[0] - self.goal_centroid[0]) + 
                            c_abs(state_centroid[1] - self.goal_centroid[1]))
            
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
                best_centroid = self.calculate_centroid(list(best_state))
                best_distance = (c_abs(best_centroid[0] - self.goal_centroid[0]) + 
                                c_abs(best_centroid[1] - self.goal_centroid[1]))
                print(f"Best block position found with centroid distance: {best_distance}")
                return self.reconstruct_path(came_from, best_state)
    
        return [self.start_state]  # No movement possible
    
    cpdef list smarter_morphing_phase(self, object start_state, double time_limit=15):
        """
        Improved Phase 2: Morph the block into the goal shape while maintaining connectivity
        Uses beam search and intelligent move generation with support for simultaneous moves
        """
        print(f"Starting Smarter Morphing Phase with {self.min_simultaneous_moves}-{self.max_simultaneous_moves} simultaneous moves...")
        cdef double start_time = time.time()
        cdef double last_improvement_time = time.time()
        
        # Initialize beam search
        open_set = [(self.improved_morphing_heuristic(start_state), 0, start_state)]
        cdef set closed_set = set()
        
        # Track path, g-scores, and best state
        cdef dict g_score = {start_state: 0}
        cdef dict came_from = {start_state: None}
        
        # Track best state seen so far
        cdef object best_state = start_state
        cdef double best_heuristic = self.improved_morphing_heuristic(start_state)
        cdef double current_heuristic, f, g, tentative_g, f_score
        cdef int iterations = 0
        cdef object current, neighbor
        
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
            
            # Check if this is the best state seen so far
            current_heuristic = self.improved_morphing_heuristic(current)
            if current_heuristic < best_heuristic:
                best_state = current
                best_heuristic = current_heuristic
                last_improvement_time = time.time()
                
                # Print progress occasionally
                if iterations % 500 == 0:
                    print(f"Progress: h={best_heuristic}, iterations={iterations}")
            
            # Check for stagnation
            if time.time() - last_improvement_time > time_limit * 0.3:
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
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.improved_morphing_heuristic(neighbor)
                    
                    # Add to open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
            
            # Beam search pruning: keep only the best states
            if len(open_set) > self.beam_width:
                open_set = heapq.nsmallest(self.beam_width, open_set)
                heapq.heapify(open_set)
        
        # If we exit the loop, either no path was found or time limit reached
        if time.time() - start_time >= time_limit:
            print(f"Morphing phase timed out after {iterations} iterations!")
        
        # Return the best state found
        return self.reconstruct_path(came_from, best_state)
    
    cpdef list reconstruct_path(self, dict came_from, object current):
        """
        Reconstruct the path from start to goal
        """
        cdef list path = []
        while current:
            path.append(list(current))
            current = came_from.get(current)
        
        path.reverse()
        return path
    
    def search(self, double time_limit=1000):
        cdef double start_time
        cdef double elapsed_time
        cdef double block_time_limit
        cdef double morphing_time_limit
        cdef list path
        cdef list block_path
        cdef list morphing_path
        cdef list block_final_state
        cdef double distance_to_goal
    
        try:
        # Record start time
            start_time = time.time()
            elapsed_time = 0
            block_time_limit = time_limit * 0.7  # Use 70% of time for block movement
            morphing_time_limit = time_limit * 0.3  # Reserve 30% for morphing
            path = []
            block_path = []
            morphing_path = []
        
            print("Starting Block Movement Phase...")
        
        # First try to move blocks close to goal without changing shape
            block_path = self.block_movement_phase(block_time_limit)
        
            if block_path:
                path.extend(block_path)
            # Get final state after block movement
                block_final_state = path[-1][1] if path else self.current_state
            
            # Calculate distance to goal centroid
                distance_to_goal = self.calculate_centroid_distance(block_final_state, self.goal_state)
            
                if distance_to_goal <= 0.1:  # If close enough to goal, we're done
                    print(f"Goal reached with block movement alone! Distance: {distance_to_goal}")
                    return path
                elif distance_to_goal <= 1.5:  # If reasonably close, try morphing
                    print(f"Block stopped {distance_to_goal} grid cells before goal centroid. Distance: {distance_to_goal}")
                    print("Starting Smarter Morphing Phase with 1-1 simultaneous moves...")
                    morphing_path = self.smarter_morphing_phase(block_final_state, morphing_time_limit)
                    if morphing_path:
                        path.extend(morphing_path)
                        return path
                    else:
                        print("Morphing phase could not find a path to goal.")
                else:
                    print(f"Best block position found with centroid distance: {distance_to_goal}")
                    print("Starting Smarter Morphing Phase with 1-1 simultaneous moves...")
                    morphing_path = self.smarter_morphing_phase(block_final_state, morphing_time_limit)
                    if morphing_path:
                        path.extend(morphing_path)
                        return path
                    else:
                        print("Morphing phase could not find a path to goal.")
            else:
                print("Block movement phase could not find a path.")
            # Try morphing directly from initial state
                print("Starting Smarter Morphing Phase from initial state...")
                morphing_path = self.smarter_morphing_phase(self.current_state, time_limit)
                if morphing_path:
                    path = morphing_path
                    return path
                else:
                    print("Morphing phase could not find a path to goal.")
        
        # If we get here, no solution was found
            print("No solution found within time limit. Returning best partial path if available.")
            return path if path else None
        
        except Exception as e:
            import traceback
            print(f"Error in search algorithm: {str(e)}")
            print(traceback.format_exc())
        # Return whatever partial path we have instead of crashing
            return []
    def visualize_path(self, list path, double interval=0.5):
        """
        Visualize the path as an animation
        """
        if not path:
            print("No path to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.ion()  # Turn on interactive mode
    
        # Get bounds for plotting
        cdef int min_x = 0
        cdef int max_x = self.grid_size[0] - 1
        cdef int min_y = 0
        cdef int max_y = self.grid_size[1] - 1
    
        # Show initial state
        ax.clear()
        ax.set_xlim(min_x - 0.5, max_x + 0.5)
        ax.set_ylim(min_y - 0.5, max_y + 0.5)
        ax.grid(True)
    
        # Draw goal positions (as outlines)
        for pos in self.goal_positions:
            rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)
    
        # Draw current positions (blue squares)
        current_positions = path[0]
        rects = []
        for pos in current_positions:
            rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='blue', alpha=0.7)
            ax.add_patch(rect)
            rects.append(rect)
        
        ax.set_title(f"Step 0/{len(path)-1}")
        plt.draw()
        plt.pause(interval)
    
        # Animate the path
        cdef int i
        cdef list new_positions
        
        for i in range(1, len(path)):
            # Update positions
            new_positions = path[i]
        
            # Clear previous positions
            for rect in rects:
                rect.remove()
        
            # Draw new positions
            rects = []
            for pos in new_positions:
                rect = plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, facecolor='blue', alpha=0.7)
                ax.add_patch(rect)
                rects.append(rect)
            
            ax.set_title(f"Step {i}/{len(path)-1}")
            plt.draw()
            plt.pause(interval)
    
        plt.ioff()  # Turn off interactive mode
        plt.show(block=True)