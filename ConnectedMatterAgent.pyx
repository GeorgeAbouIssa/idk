import heapq
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

cdef class ConnectedMatterAgent:
    cdef public tuple grid_size
    cdef public list start_positions
    cdef public list goal_positions
    cdef public str topology
    cdef public int max_simultaneous_moves
    cdef public int min_simultaneous_moves
    cdef public list directions
    cdef public object start_state
    cdef public object goal_state
    cdef public tuple goal_centroid
    cdef public dict valid_moves_cache
    cdef public dict articulation_points_cache
    cdef public dict connectivity_check_cache
    cdef public int beam_width
    cdef public int max_iterations
    cdef public object target_state  # Subset of blocks to be used for goal
    cdef public object non_target_state  # Blocks that won't move to goal
    cdef public bint allow_disconnection  # Flag to allow blocks to disconnect
    cdef public list target_block_list  # List of blocks that will be moved
    cdef public list fixed_block_list  # List of blocks that won't be moved

    def __init__(self, tuple grid_size, list start_positions, list goal_positions, str topology="moore", 
                 int max_simultaneous_moves=1, int min_simultaneous_moves=1):
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
        
        # Ensure no duplicates in goal positions
        unique_goal_positions = []
        goal_positions_set = set()
        for pos in goal_positions:
            if pos not in goal_positions_set:
                goal_positions_set.add(pos)
                unique_goal_positions.append(pos)
                
        if len(unique_goal_positions) != len(goal_positions):
            print("WARNING: Duplicate positions detected in goal state. Removed duplicates.")
            self.goal_positions = unique_goal_positions
            
        # Initialize the start and goal states
        self.start_state = frozenset((x, y) for x, y in self.start_positions)
        self.goal_state = frozenset((x, y) for x, y in self.goal_positions)
        
        # Calculate the centroid of the goal positions for block movement phase
        self.goal_centroid = self.calculate_centroid(self.goal_positions)
        
        # Handle cases where goal has fewer blocks than start
        if len(self.goal_positions) < len(start_positions):
            # Flag to allow disconnection when goal has fewer blocks
            self.allow_disconnection = True
            
            # Select the subset of blocks closest to the goal centroid
            self.target_block_list = self.select_closest_blocks_to_goal()
            self.fixed_block_list = [pos for pos in self.start_positions if pos not in self.target_block_list]
            
            # Convert to frozensets for state operations
            self.target_state = frozenset((x, y) for x, y in self.target_block_list)
            self.non_target_state = frozenset((x, y) for x, y in self.fixed_block_list)
            
            print(f"Goal has fewer blocks ({len(self.goal_positions)}) than start ({len(start_positions)})")
            print(f"Selected {len(self.target_block_list)} blocks closest to the goal centroid")
            print(f"Blocks will be allowed to disconnect during movement")
            print(f"Fixed blocks: {len(self.fixed_block_list)} will remain stationary")
        else:
            # If goal has same or more blocks, all start blocks are target blocks
            self.allow_disconnection = False
            self.target_block_list = self.start_positions.copy()
            self.fixed_block_list = []
            self.target_state = self.start_state
            self.non_target_state = frozenset()
        
        # Cache for valid moves to avoid recomputation
        self.valid_moves_cache = {}
        
        # For optimizing the search
        self.articulation_points_cache = {}
        self.connectivity_check_cache = {}
        
        # Enhanced parameters for improved search
        self.beam_width = 500  # Increased beam width for better exploration
        self.max_iterations = 10000  # Limit iterations to prevent infinite loops
        
    def select_closest_blocks_to_goal(self):
        """
        Select blocks from start state that are closest to the goal centroid
        Returns a list of selected blocks
        """
        # Calculate distances from each start position to goal centroid
        distances = []
        for pos in self.start_positions:
            # Manhattan distance to centroid
            dist = abs(pos[0] - self.goal_centroid[0]) + abs(pos[1] - self.goal_centroid[1])
            distances.append((dist, pos))
        
        # Sort by distance (ascending)
        distances.sort()
        
        # Select the number of blocks needed for the goal
        selected_blocks = [pos for _, pos in distances[:len(self.goal_positions)]]
        
        return selected_blocks
        
    def calculate_centroid(self, positions):
        """Calculate the centroid (average position) of a set of positions"""
        cdef double x_sum, y_sum
        
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
    
    def is_target_connected(self, state):
        """
        Check if target blocks in a state are connected to each other
        This is used when allow_disconnection is True
        """
        if not self.allow_disconnection:
            return self.is_connected(state)
            
        # Extract only target blocks
        target_blocks = [pos for pos in state if pos not in self.non_target_state]
        
        # Check connectivity of just the target blocks
        return self.is_connected(target_blocks)
    
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
            cdef int children = 0
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
    
    def get_target_articulation_points(self, state):
        """
        Find articulation points in just the target blocks
        This is used when allow_disconnection is True
        """
        if not self.allow_disconnection:
            return self.get_articulation_points(state)
            
        # Extract only target blocks
        target_blocks = set(pos for pos in state if pos not in self.non_target_state)
        
        # Get articulation points of just the target blocks
        return self.get_articulation_points(target_blocks)
    
    def has_overlapping_blocks(self, state):
        """Check if a state has any overlapping blocks"""
        state_list = list(state)
        return len(state_list) != len(set(state_list))
        
    def get_valid_block_moves(self, state):
        """
        Generate valid moves for blocks
        A valid block move shifts the target elements in the same direction
        Fixed blocks remain stationary
        """
        valid_moves = []
        
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            return valid_moves
            
        # Extract movable blocks (target blocks)
        state_list = list(state)
        
        # If there are fixed blocks, they should remain stationary
        if self.allow_disconnection and self.non_target_state:
            # Only move target blocks
            movable_blocks = [pos for pos in state_list if pos not in self.non_target_state]
            fixed_blocks = [pos for pos in state_list if pos in self.non_target_state]
        else:
            # Move all blocks
            movable_blocks = state_list
            fixed_blocks = []
        
        # Try moving the movable blocks in each direction
        for dx, dy in self.directions:
            # Calculate new positions after moving
            new_positions = [(pos[0] + dx, pos[1] + dy) for pos in movable_blocks]
            
            # Check if all new positions are valid:
            # 1. Within bounds
            # 2. Not occupied by fixed blocks
            # 3. No position duplication (no overlap)
            all_valid = True
            new_pos_set = set()
            
            for pos in new_positions:
                # Check bounds
                if not (0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]):
                    all_valid = False
                    break
                    
                # Check collision with fixed blocks
                if pos in fixed_blocks:
                    all_valid = False
                    break
                    
                # Check for duplicates (overlap)
                if pos in new_pos_set:
                    all_valid = False
                    break
                    
                new_pos_set.add(pos)
            
            # Only consider moves that keep all positions valid
            if all_valid:
                # Combine with fixed blocks to create the new state
                new_state = frozenset(new_positions + fixed_blocks)
                
                # For connectivity check, if we're allowing disconnection, we only
                # care about target blocks staying connected to each other
                if self.allow_disconnection:
                    # Only check target block connectivity
                    if self.is_target_connected(new_state):
                        valid_moves.append(new_state)
                else:
                    # Regular connectivity check for all blocks
                    if self.is_connected(new_state):
                        valid_moves.append(new_state)
        
        return valid_moves
    
    def get_valid_morphing_moves(self, state):
        """
        Generate valid morphing moves
        Supports multiple simultaneous block movements with minimum requirement
        Fixed blocks remain stationary, only target blocks move
        """
        state_key = hash(state)
        if state_key in self.valid_moves_cache:
            return self.valid_moves_cache[state_key]
            
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            self.valid_moves_cache[state_key] = []
            return []
            
        # Get single block moves first
        single_moves = []
        state_set = set(state)
        
        # Identify fixed blocks vs movable blocks
        fixed_blocks = set()
        if self.allow_disconnection:
            fixed_blocks = state_set.intersection(self.non_target_state)
        
        # If there are fixed blocks, they should remain stationary
        if self.allow_disconnection and fixed_blocks:
            # Only consider target blocks as movable
            movable_candidates = state_set - fixed_blocks
            
            # Get articulation points among just the target blocks
            target_blocks = [pos for pos in state_set if pos not in fixed_blocks]
            if target_blocks:
                target_articulation_points = self.get_articulation_points(set(target_blocks))
                movable_points = set(target_blocks) - target_articulation_points
                
                # If all target points are critical, try moving one anyway but verify connectivity
                if not movable_points and target_articulation_points:
                    for point in target_articulation_points:
                        # Try removing and see if structure remains connected
                        temp_target_blocks = set(target_blocks)
                        temp_target_blocks.remove(point)
                        if self.is_connected(temp_target_blocks) or len(temp_target_blocks) <= 1:
                            movable_points.add(point)
            else:
                movable_points = set()
        else:
            # Standard connectivity rules apply to all blocks
            # Find non-critical points that can move without breaking connectivity
            articulation_points = self.get_articulation_points(state_set)
            movable_points = state_set - articulation_points
            
            # If all points are critical, try moving one anyway but verify connectivity 
            if not movable_points and articulation_points:
                for point in articulation_points:
                    # Try removing and see if structure remains connected
                    temp_state = state_set.copy()
                    temp_state.remove(point)
                    if self.is_connected(temp_state):
                        movable_points.add(point)
        
        # Generate single block moves
        for point in movable_points:
            # Skip fixed blocks
            if point in fixed_blocks:
                continue
                
            # Try moving in each direction
            for dx, dy in self.directions:
                new_pos = (point[0] + dx, point[1] + dy)
                
                # Skip if out of bounds
                if not (0 <= new_pos[0] < self.grid_size[0] and 
                        0 <= new_pos[1] < self.grid_size[1]):
                    continue
                
                # Skip if already occupied
                if new_pos in state_set:
                    continue
                
                # Create new state by moving the point
                new_state_set = state_set.copy()
                new_state_set.remove(point)
                new_state_set.add(new_pos)
                
                # Check for overlapping positions
                if len(new_state_set) != len(state_set):
                    continue
                
                if self.allow_disconnection:
                    # With disconnection allowed, we only need to check:
                    # 1. Target blocks stay connected to each other
                    # 2. No overlap between target and fixed blocks
                    
                    # Extract target blocks
                    new_target_blocks = [pos for pos in new_state_set if pos not in self.non_target_state]
                    
                    # Skip if the number of target blocks doesn't match the goal size
                    if len(new_target_blocks) != len(self.goal_positions):
                        continue
                    
                    # Ensure no overlaps between target blocks
                    if len(new_target_blocks) != len(set(new_target_blocks)):
                        continue
                    
                    # Check if target blocks collide with fixed blocks
                    new_fixed_blocks = [pos for pos in new_state_set if pos in self.non_target_state]
                    if any(pos in new_fixed_blocks for pos in new_target_blocks):
                        continue
                        
                    # Check target block connectivity
                    if self.is_connected(new_target_blocks) or len(new_target_blocks) <= 1:
                        single_moves.append((point, new_pos))
                else:
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
        
        # Generate multi-block moves
        for k in range(self.min_simultaneous_moves, min(self.max_simultaneous_moves + 1, len(single_moves) + 1)):
            # Generate combinations of k moves
            for combo in self._generate_move_combinations(single_moves, k):
                # Check if the combination is valid (no conflicts)
                if self._is_valid_move_combination(combo, state_set):
                    # Apply the combination
                    new_state = self._apply_moves(state_set, combo)
                    
                    # Check for overlapping positions
                    if self.has_overlapping_blocks(new_state):
                        continue
                    
                    # Additional validation for goal with fewer blocks
                    if self.allow_disconnection:
                        # Extract target and fixed blocks from the new state
                        new_target_blocks = [pos for pos in new_state if pos not in self.non_target_state]
                        new_fixed_blocks = [pos for pos in new_state if pos in self.non_target_state]
                        
                        # Skip if any target block occupies the same position as a fixed block
                        if any(pos in new_fixed_blocks for pos in new_target_blocks):
                            continue
                        
                        # Skip if the number of target blocks doesn't match the goal size
                        if len(new_target_blocks) != len(self.goal_positions):
                            continue
                        
                        # Ensure no overlaps among target blocks
                        if len(new_target_blocks) != len(set(new_target_blocks)):
                            continue
                            
                        # Check if target blocks remain connected to each other
                        if self.is_connected(new_target_blocks) or len(new_target_blocks) <= 1:
                            valid_moves.append(frozenset(new_state))
                    else:
                        # Check full connectivity for standard goal
                        if self.is_connected(new_state):
                            valid_moves.append(frozenset(new_state))
        
        # If no valid moves with min_simultaneous_moves, fallback to single moves if allowed
        if not valid_moves and self.min_simultaneous_moves == 1:
            valid_moves = []
            for move in single_moves:
                new_state = self._apply_moves(state_set, [move])
                
                # Skip states with overlapping blocks
                if self.has_overlapping_blocks(new_state):
                    continue
                    
                # Additional validation for goal with fewer blocks
                if self.allow_disconnection:
                    # Extract target and fixed blocks from the new state
                    new_target_blocks = [pos for pos in new_state if pos not in self.non_target_state]
                    new_fixed_blocks = [pos for pos in new_state if pos in self.non_target_state]
                    
                    # Skip if any target block occupies the same position as a fixed block
                    if any(pos in new_fixed_blocks for pos in new_target_blocks):
                        continue
                    
                    # Skip if the number of target blocks doesn't match the goal size
                    if len(new_target_blocks) != len(self.goal_positions):
                        continue
                    
                    # Ensure no overlaps among target blocks
                    if len(new_target_blocks) != len(set(new_target_blocks)):
                        continue
                        
                    # Check if target blocks remain connected to each other
                    if self.is_connected(new_target_blocks) or len(new_target_blocks) <= 1:
                        valid_moves.append(frozenset(new_state))
                else:
                    # Check full connectivity for standard goal
                    if self.is_connected(new_state):
                        valid_moves.append(frozenset(new_state))
        
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
        
        # Additional check: Make sure targets don't collide with unmoved blocks
        # Identify blocks that won't be moving in this step
        remaining_blocks = state_set - sources
        
        # Check that no target position collides with a block that isn't moving
        for tgt in targets:
            if tgt in remaining_blocks:
                return False
        
        return True
    
    def _apply_moves(self, state_set, moves):
        """Apply a list of moves to the state"""
        new_state = state_set.copy()
        for src, tgt in moves:
            new_state.remove(src)
            new_state.add(tgt)
        return new_state
    
    def get_smart_chain_moves(self, state):
        """
        Generate chain moves where one block moves into the space of another
        while that block moves elsewhere
        Fixed blocks remain stationary
        """
        cdef double min_dist
        cdef int dx, dy
        
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            return []
            
        state_set = set(state)
        valid_moves = []
        
        # Identify fixed blocks
        fixed_blocks = set()
        if self.allow_disconnection:
            fixed_blocks = state_set.intersection(self.non_target_state)
        
        # For each movable block, try to move it toward a goal position
        for pos in state_set:
            # Skip fixed blocks
            if pos in fixed_blocks:
                continue
                
            # Find closest goal position
            min_dist = float('inf')
            closest_goal = None
            
            for goal_pos in self.goal_state:
                if goal_pos not in state_set:  # Only consider unoccupied goals
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
            
            # Skip if out of bounds
            if not (0 <= next_pos[0] < self.grid_size[0] and 
                    0 <= next_pos[1] < self.grid_size[1]):
                continue
            
            # If next position is occupied, try chain move
            if next_pos in state_set:
                # Skip if the occupied position is a fixed block
                if next_pos in fixed_blocks:
                    continue
                    
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
                    
                    # Check for overlapping positions
                    if len(new_state_set) != len(state_set):
                        continue
                    
                    # Handle connectivity based on goal type
                    if self.allow_disconnection:
                        # Check connectivity among target blocks only
                        new_target_blocks = [p for p in new_state_set if p not in self.non_target_state]
                        
                        # Extract target and fixed blocks from the new state
                        new_fixed_blocks = [p for p in new_state_set if p in self.non_target_state]
                        
                        # Skip if any target block occupies the same position as a fixed block
                        if any(p in new_fixed_blocks for p in new_target_blocks):
                            continue
                        
                        # Check target block connectivity
                        if self.is_connected(new_target_blocks) or len(new_target_blocks) <= 1:
                            valid_moves.append(frozenset(new_state_set))
                    else:
                        # Standard connectivity for all blocks
                        if self.is_connected(new_state_set):
                            valid_moves.append(frozenset(new_state_set))
            
            # If next position is unoccupied, try direct move
            else:
                new_state_set = state_set.copy()
                new_state_set.remove(pos)
                new_state_set.add(next_pos)
                
                # Check for overlapping positions
                if len(new_state_set) != len(state_set):
                    continue
                
                # Handle connectivity based on goal type
                if self.allow_disconnection:
                    # Check connectivity among target blocks only
                    new_target_blocks = [p for p in new_state_set if p not in self.non_target_state]
                    
                    # Extract target and fixed blocks from the new state
                    new_fixed_blocks = [p for p in new_state_set if p in self.non_target_state]
                    
                    # Skip if any target block occupies the same position as a fixed block
                    if any(p in new_fixed_blocks for p in new_target_blocks):
                        continue
                    
                    # Check target block connectivity
                    if self.is_connected(new_target_blocks) or len(new_target_blocks) <= 1:
                        valid_moves.append(frozenset(new_state_set))
                else:
                    # Standard connectivity for all blocks
                    if self.is_connected(new_state_set):
                        valid_moves.append(frozenset(new_state_set))
        
        return valid_moves
    
    def get_sliding_chain_moves(self, state):
        """
        Generate sliding chain moves where multiple blocks move in sequence
        to navigate tight spaces
        Fixed blocks remain stationary
        """
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            return []
            
        state_set = set(state)
        valid_moves = []
        
        # Identify fixed blocks
        fixed_blocks = set()
        if self.allow_disconnection:
            fixed_blocks = state_set.intersection(self.non_target_state)
        
        # For each movable block, try to initiate a sliding chain
        for pos in state_set:
            # Skip fixed blocks
            if pos in fixed_blocks:
                continue
                
            # Skip if it's a critical articulation point (unless disconnection allowed)
            if not self.allow_disconnection:
                articulation_points = self.get_articulation_points(state_set)
                if pos in articulation_points and len(articulation_points) <= 20:
                    continue
            elif self.allow_disconnection:
                # When allowing disconnection, check articulation points only among target blocks
                target_blocks = [p for p in state_set if p not in fixed_blocks]
                if target_blocks:
                    target_articulation_points = self.get_articulation_points(set(target_blocks))
                    if pos in target_articulation_points and len(target_articulation_points) <= 15:
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
                        
                        # Check for overlapping positions
                        if len(new_state_set) != len(state_set):
                            continue
                        
                        # Handle connectivity based on goal type
                        if self.allow_disconnection:
                            # Check connectivity among target blocks only
                            new_target_blocks = [p for p in new_state_set if p not in self.non_target_state]
                            
                            # Extract target and fixed blocks from the new state
                            new_fixed_blocks = [p for p in new_state_set if p in self.non_target_state]
                            
                            # Skip if any target block occupies the same position as a fixed block
                            if any(p in new_fixed_blocks for p in new_target_blocks):
                                continue
                            
                            # Check target block connectivity
                            if self.is_connected(new_target_blocks) or len(new_target_blocks) <= 1:
                                valid_moves.append(frozenset(new_state_set))
                        else:
                            # Standard connectivity for all blocks
                            if self.is_connected(new_state_set):
                                valid_moves.append(frozenset(new_state_set))
                        
                        # No need to continue if we can't reach this position
                        break
        
        return valid_moves
    
    def get_all_valid_moves(self, state):
        """
        Combine all move generation methods to maximize options
        """
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            return []
            
        # Start with basic morphing moves
        basic_moves = self.get_valid_morphing_moves(state)
        
        # Add chain moves
        chain_moves = self.get_smart_chain_moves(state)
        
        # Add sliding chain moves
        sliding_moves = self.get_sliding_chain_moves(state)
        
        # Combine all moves (frozensets automatically handle duplicates)
        all_moves = list(set(basic_moves + chain_moves + sliding_moves))
        
        return all_moves
    
    def block_heuristic(self, state):
        """
        Heuristic for block movement phase:
        Calculate Manhattan distance from current target blocks to goal centroid
        """
        if not state:
            return float('inf')
        
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            return float('inf')
            
        if self.allow_disconnection:
            # Extract target blocks from current state
            target_blocks = [pos for pos in state if pos not in self.non_target_state]
            if not target_blocks:
                return float('inf')
                
            # Calculate centroid of target blocks only
            target_centroid = self.calculate_centroid(target_blocks)
            
            # Pure Manhattan distance between centroids
            return abs(target_centroid[0] - self.goal_centroid[0]) + abs(target_centroid[1] - self.goal_centroid[1])
        else:
            # Standard centroid calculation for all blocks
            current_centroid = self.calculate_centroid(state)
            
            # Pure Manhattan distance between centroids
            return abs(current_centroid[0] - self.goal_centroid[0]) + abs(current_centroid[1] - self.goal_centroid[1])
    
    def improved_morphing_heuristic(self, state):
        """
        Improved heuristic for morphing phase:
        Uses bipartite matching to find optimal assignment of blocks to goal positions
        When goal has fewer blocks, only consider target blocks
        """
        cdef double total_distance = 0
        cdef double min_dist
        cdef int best_j
        cdef int matching_positions
        cdef double connectivity_bonus
        
        if not state:
            return float('inf')
            
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(state):
            return float('inf')
        
        # If we're targeting a subset of blocks (goal has fewer blocks)
        if self.allow_disconnection:
            # Extract target blocks (exclude fixed blocks)
            target_blocks = [pos for pos in state if pos not in self.non_target_state]
            
            # If number of target blocks doesn't match goal positions, this shouldn't happen
            if len(target_blocks) != len(self.goal_positions):
                return float('inf')
                
            goal_list = list(self.goal_state)
            
            # Build distance matrix only for target blocks
            distance_matrix = []
            for pos in target_blocks:
                row = []
                for goal_pos in goal_list:
                    # Manhattan distance
                    dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                    row.append(dist)
                distance_matrix.append(row)
            
            # Greedy assignment
            assigned_cols = set()
            
            for i in range(len(target_blocks)):
                # Find closest unassigned goal position
                min_dist = float('inf')
                best_j = -1
                
                for j in range(len(goal_list)):
                    if j not in assigned_cols and distance_matrix[i][j] < min_dist:
                        min_dist = distance_matrix[i][j]
                        best_j = j
                
                if best_j != -1:
                    assigned_cols.add(best_j)
                    total_distance += min_dist
                else:
                    # No assignment possible
                    return float('inf')
            
            # Add connectivity bonus: prefer states that have more blocks in goal positions
            matching_positions = sum(1 for pos in target_blocks if pos in self.goal_state)
            connectivity_bonus = -matching_positions * 0.5  # Negative to encourage more matches
            
            return total_distance + connectivity_bonus
            
        else:
            # Original logic for when goal has same number of blocks as start
            state_list = list(state)
            goal_list = list(self.goal_state)
            
            # Build distance matrix
            distances = []
            for pos in state_list:
                row = []
                for goal_pos in goal_list:
                    # Manhattan distance
                    dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                    row.append(dist)
                distances.append(row)
            
            # Use greedy assignment algorithm
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
                else:
                    # No assignment possible
                    return float('inf')
            
            # Add connectivity bonus: prefer states that have more blocks in goal positions
            matching_positions = len(state.intersection(self.goal_state))
            connectivity_bonus = -matching_positions * 0.5  # Negative to encourage more matches
            
            return total_distance + connectivity_bonus
    
    def block_movement_phase(self, double time_limit=15):
        """
        Phase 1: Move blocks toward the goal centroid
        If goal has fewer blocks than start, only move the target blocks
        Fixed blocks remain stationary throughout
        """
        cdef double start_time
        cdef double min_distance = 1.0
        cdef double max_distance = 1.0
        cdef double centroid_distance, neighbor_distance, distance_penalty
        cdef double adjusted_heuristic, f_score, best_distance_diff, distance, distance_diff, best_distance
        cdef int tentative_g
        
        print("Starting Block Movement Phase...")
        if self.allow_disconnection:
            print(f"Moving only {len(self.target_block_list)} target blocks, keeping {len(self.fixed_block_list)} blocks stationary")
        
        start_time = time.time()

        # Create initial state - if disconnection is allowed, we need to include fixed blocks
        initial_state = self.start_state

        # Initialize A* search
        open_set = [(self.block_heuristic(initial_state), 0, initial_state)]
        closed_set = set()

        # Track path and g-scores
        g_score = {initial_state: 0}
        came_from = {initial_state: None}

        while open_set and time.time() - start_time < time_limit:
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
    
            # Skip if already processed
            if current in closed_set:
                continue
            
            # Skip states with overlapping blocks
            if self.has_overlapping_blocks(current):
                continue
        
            # Check if we're at the desired distance from the goal centroid
            if self.allow_disconnection:
                # Extract target blocks
                target_blocks = [pos for pos in current if pos not in self.non_target_state]
                
                # Calculate centroid of target blocks only
                if target_blocks:
                    target_centroid = self.calculate_centroid(target_blocks)
                    centroid_distance = (abs(target_centroid[0] - self.goal_centroid[0]) + 
                                        abs(target_centroid[1] - self.goal_centroid[1]))
                else:
                    centroid_distance = float('inf')
            else:
                # Standard centroid calculation
                current_centroid = self.calculate_centroid(current)
                centroid_distance = (abs(current_centroid[0] - self.goal_centroid[0]) + 
                                    abs(current_centroid[1] - self.goal_centroid[1]))
                        
            if min_distance <= centroid_distance <= max_distance:
                print(f"Blocks stopped 1 grid cell before goal centroid. Distance: {centroid_distance}")
                return self.reconstruct_path(came_from, current)
        
            closed_set.add(current)
    
            # Process neighbor states (block moves)
            for neighbor in self.get_valid_block_moves(current):
                # Skip states with overlapping blocks
                if self.has_overlapping_blocks(neighbor):
                    continue
                    
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g-score
                tentative_g = g_score[current] + 1
        
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This is a better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                
                    # Adjusted heuristic to prefer states near but not at centroid
                    if self.allow_disconnection:
                        # Extract target blocks for distance calculation
                        neighbor_target_blocks = [pos for pos in neighbor if pos not in self.non_target_state]
                        if neighbor_target_blocks:
                            neighbor_centroid = self.calculate_centroid(neighbor_target_blocks)
                            neighbor_distance = (abs(neighbor_centroid[0] - self.goal_centroid[0]) + 
                                               abs(neighbor_centroid[1] - self.goal_centroid[1]))
                        else:
                            # This shouldn't happen with our algorithm
                            neighbor_distance = float('inf')
                    else:
                        # Standard centroid calculation
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
                # Skip states with overlapping blocks
                if self.has_overlapping_blocks(state):
                    continue
                    
                if self.allow_disconnection:
                    # Calculate distance for target blocks only
                    target_blocks = [pos for pos in state if pos not in self.non_target_state]
                    if target_blocks:
                        state_centroid = self.calculate_centroid(target_blocks)
                        distance = (abs(state_centroid[0] - self.goal_centroid[0]) + 
                                  abs(state_centroid[1] - self.goal_centroid[1]))
                    else:
                        distance = float('inf')
                else:
                    # Standard centroid calculation
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
                if self.allow_disconnection:
                    # Calculate distance for target blocks only
                    target_blocks = [pos for pos in best_state if pos not in self.non_target_state]
                    if target_blocks:
                        best_centroid = self.calculate_centroid(target_blocks)
                        best_distance = (abs(best_centroid[0] - self.goal_centroid[0]) + 
                                       abs(best_centroid[1] - self.goal_centroid[1]))
                    else:
                        best_distance = float('inf')
                else:
                    # Standard centroid calculation
                    best_centroid = self.calculate_centroid(best_state)
                    best_distance = (abs(best_centroid[0] - self.goal_centroid[0]) + 
                                   abs(best_centroid[1] - self.goal_centroid[1]))
                                   
                print(f"Best block position found with centroid distance: {best_distance}")
                return self.reconstruct_path(came_from, best_state)
    
        return [self.start_state]  # No movement possible
    
    def smarter_morphing_phase(self, start_state, double time_limit=15):
        """
        Improved Phase 2: Morph the blocks into the goal shape
        With disconnection allowed, only target blocks are morphed
        Fixed blocks remain stationary
        """
        cdef double start_time
        cdef double best_heuristic, current_heuristic, last_improvement_time, f_score
        cdef int iterations = 0
        cdef int tentative_g
    
        # Skip states with overlapping blocks
        if self.has_overlapping_blocks(start_state):
            print("WARNING: Starting state for morphing has overlapping blocks!")
            return [start_state]
        
        print(f"Starting Smarter Morphing Phase with {self.min_simultaneous_moves}-{self.max_simultaneous_moves} simultaneous moves...")
        if self.allow_disconnection:
            print(f"Morphing only {len(self.target_block_list)} target blocks, keeping {len(self.fixed_block_list)} blocks stationary")
    
        start_time = time.time()
    
        # Initialize beam search
        open_set = [(self.improved_morphing_heuristic(start_state), 0, start_state)]
        closed_set = set()
    
        # Track path, g-scores, and best state
        g_score = {start_state: 0}
        came_from = {start_state: None}
    
        # Track best state seen so far
        best_state = start_state
        best_heuristic = self.improved_morphing_heuristic(start_state)
        last_improvement_time = time.time()
    
        # Determine whether we're targeting a subset of blocks
        targeting_subset = self.allow_disconnection
    
        while open_set and time.time() - start_time < time_limit:
            iterations += 1
        
            # Get state with lowest f-score
            f, g, current = heapq.heappop(open_set)
        
            # Skip if already processed
            if current in closed_set:
                continue
        
            # Skip states with overlapping blocks
            if self.has_overlapping_blocks(current):
                continue
            
            # Check if goal reached
            if targeting_subset:
                # For subset targeting, check if target blocks match goal positions
                # Extract target blocks from current state (exclude fixed blocks)
                target_blocks = frozenset(pos for pos in current if pos not in self.non_target_state)
            
                # Check if all goal positions are filled by target blocks
                if self.goal_state.issubset(target_blocks) or target_blocks.issubset(self.goal_state):
                    print(f"Goal approximation reached after {iterations} iterations!")
                    return self.reconstruct_path(came_from, current)
                
                # Alternative goal check: if enough blocks are in the right positions
                matching_positions = len(target_blocks.intersection(self.goal_state))
                if matching_positions == len(self.goal_state):
                    print(f"Goal positions matched after {iterations} iterations!")
                    return self.reconstruct_path(came_from, current)
            else:
                # For exact matching, use original goal check
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
                # Skip states with overlapping blocks
                if self.has_overlapping_blocks(neighbor):
                    continue
                
                # Additional validation for states with fewer blocks in goal
                if self.allow_disconnection:
                    # Extract only target blocks
                    target_blocks = [pos for pos in neighbor if pos not in self.non_target_state]
                
                    # Skip if we don't have the right number of target blocks
                    if len(target_blocks) != len(self.goal_positions):
                        continue
                    
                    # Skip if target blocks overlap with each other
                    if len(target_blocks) != len(set(target_blocks)):
                        continue
                
                    # Skip if any target block occupies the same position as a fixed block
                    fixed_blocks = [pos for pos in neighbor if pos in self.non_target_state]
                    if any(pos in fixed_blocks for pos in target_blocks):
                        continue
                        
                    # Check if target blocks remain connected to each other
                    if not (self.is_connected(target_blocks) or len(target_blocks) <= 1):
                        continue
            
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
    
    def reconstruct_path(self, came_from, current):
        """
        Reconstruct the path from start to goal
        """
        path = []
        while current:
            # Convert frozenset to list
            if isinstance(current, frozenset):
                path.append(list(current))
            else:
                path.append(list(current))
                
            current = came_from.get(current)
        
        path.reverse()
        return path
    
    def search(self, double time_limit=30):
        """
        Main search method combining block movement and smarter morphing
        """
        # Allocate time for each phase
        cdef double block_time_limit = time_limit * 0.3  # 30% for block movement
        cdef double morphing_time_limit = time_limit * 0.7  # 70% for morphing
        
        # Phase 1: Block Movement
        block_path = self.block_movement_phase(block_time_limit)
        
        if not block_path:
            print("Block movement phase failed!")
            return None
        
        # Verify the final state from block movement has no overlapping blocks
        block_final_list = block_path[-1]
        if len(block_final_list) != len(set(block_final_list)):
            print("WARNING: Block movement produced a state with overlapping blocks!")
            # Try to fix it by removing duplicates
            block_final_list = list(set(block_final_list))
            block_path[-1] = block_final_list
        
        # Get the final state from block movement phase
        block_final_state = frozenset(block_final_list)
        
        # Phase 2: Smarter Morphing
        morphing_path = self.smarter_morphing_phase(block_final_state, morphing_time_limit)
        
        if not morphing_path:
            print("Morphing phase failed!")
            return block_path
        
        # Combine paths (remove duplicate state at transition)
        combined_path = block_path[:-1] + morphing_path
        
        # Final check for overlapping blocks in any state
        for i, state in enumerate(combined_path):
            if len(state) != len(set(state)):
                print(f"WARNING: State {i} in path has overlapping blocks!")
                # We could fix it, but that would alter the path - let's leave it as a warning
        
        return combined_path
    
    def visualize_path(self, path, double interval=0.5):
        """
        Visualize the path as an animation
        """
        cdef int min_x = 0
        cdef int max_x = self.grid_size[0] - 1
        cdef int min_y = 0
        cdef int max_y = self.grid_size[1] - 1
        
        if not path:
            print("No path to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.ion()  # Turn on interactive mode
    
        # Show initial state
        ax.clear()
        ax.set_xlim(min_x - 0.5, max_x + 0.5)
        ax.set_ylim(min_y - 0.5, max_y + 0.5)
        ax.grid(True)
    
        # Draw goal positions (as outlines)
        for pos in self.goal_positions:
            rect = plt.Rectangle((pos[1], pos[0]), 1, 1, fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)
    
        # Draw current positions (blue squares)
        current_positions = path[0]
        rects = []
        
        # Check for duplicates
        if len(current_positions) != len(set(current_positions)):
            print("WARNING: Initial state has overlapping blocks!")
        
        for pos in current_positions:
            if pos in self.fixed_block_list:
                # Fixed blocks in light blue
                rect = plt.Rectangle((pos[1], pos[0]), 1, 1, facecolor='lightblue', alpha=0.7)
            else:
                # Target blocks in blue
                rect = plt.Rectangle((pos[1], pos[0]), 1, 1, facecolor='blue', alpha=0.7)
            ax.add_patch(rect)
            rects.append(rect)
        
        ax.set_title(f"Step 0/{len(path)-1}")
        plt.draw()
        plt.pause(interval)
    
        # Animate the path
        for i in range(1, len(path)):
            # Update positions
            new_positions = path[i]
            
            # Check for duplicates
            if len(new_positions) != len(set(new_positions)):
                print(f"WARNING: State {i} has overlapping blocks!")
        
            # Clear previous positions
            for rect in rects:
                rect.remove()
        
            # Draw new positions
            rects = []
            for pos in new_positions:
                if pos in self.fixed_block_list:
                    # Fixed blocks in light blue
                    rect = plt.Rectangle((pos[1], pos[0]), 1, 1, facecolor='lightblue', alpha=0.7)
                else:
                    # Target blocks in blue
                    rect = plt.Rectangle((pos[1], pos[0]), 1, 1, facecolor='blue', alpha=0.7)
                ax.add_patch(rect)
                rects.append(rect)
            
            ax.set_title(f"Step {i}/{len(path)-1}")
            plt.draw()
            plt.pause(interval)
    
        plt.ioff()  # Turn off interactive mode
        plt.show(block=True)