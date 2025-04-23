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
    cdef public list goal_components  # List of connected components in goal
    cdef public list goal_component_centroids  # Centroids for each goal component
    cdef public dict block_component_assignment  # Maps target blocks to goal components
    cdef public bint multi_component_goal  # Flag indicating goal has multiple components

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
        
        # Analyze the goal state for connected components
        self.goal_components = self.find_connected_components(self.goal_positions)
        self.multi_component_goal = len(self.goal_components) > 1
        
        if self.multi_component_goal:
            print(f"Goal has {len(self.goal_components)} disconnected components")
            # Calculate centroid for each component
            self.goal_component_centroids = [self.calculate_centroid(component) 
                                            for component in self.goal_components]
            for i, (component, centroid) in enumerate(zip(self.goal_components, self.goal_component_centroids)):
                print(f"Component {i+1}: {len(component)} blocks, centroid at {centroid}")
        
        # Calculate the centroid of all goal positions (still useful for overall movement)
        self.goal_centroid = self.calculate_centroid(self.goal_positions)
        
        # Handle cases where goal has fewer blocks than start
        if len(self.goal_positions) < len(start_positions):
            # Flag to allow disconnection when goal has fewer blocks
            self.allow_disconnection = True
            
            if self.multi_component_goal:
                # Select blocks for each component separately based on proximity to component centroids
                self.target_block_list = self.select_blocks_for_components()
            else:
                # Standard selection for single-component goals
                self.target_block_list = self.select_closest_blocks_to_goal()
                
            self.fixed_block_list = [pos for pos in self.start_positions if pos not in self.target_block_list]
            
            # Convert to frozensets for state operations
            self.target_state = frozenset((x, y) for x, y in self.target_block_list)
            self.non_target_state = frozenset((x, y) for x, y in self.fixed_block_list)
            
            print(f"Goal has fewer blocks ({len(self.goal_positions)}) than start ({len(start_positions)})")
            print(f"Selected {len(self.target_block_list)} blocks closest to the goal")
            print(f"Blocks will be allowed to disconnect during movement")
            print(f"Fixed blocks: {len(self.fixed_block_list)} will remain stationary")
        else:
            # If goal has same or more blocks, all start blocks are target blocks
            self.allow_disconnection = False
            self.target_block_list = self.start_positions.copy()
            self.fixed_block_list = []
            self.target_state = self.start_state
            self.non_target_state = frozenset()
            # No component assignment needed when all blocks are targets
            self.block_component_assignment = {}
        
        # Cache for valid moves to avoid recomputation
        self.valid_moves_cache = {}
        
        # For optimizing the search
        self.articulation_points_cache = {}
        self.connectivity_check_cache = {}
        
        # Enhanced parameters for improved search
        self.beam_width = 500  # Increased beam width for better exploration
        self.max_iterations = 10000  # Limit iterations to prevent infinite loops
    
    def find_connected_components(self, positions):
        """Find all connected components in a set of positions"""
        if not positions:
            return []
            
        positions_set = set(positions)
        components = []
        
        while positions_set:
            # Take an unvisited position as the start of a new component
            start = next(iter(positions_set))
            
            # Find all positions connected to start using BFS
            component = set()
            visited = {start}
            queue = deque([start])
            
            while queue:
                current = queue.popleft()
                component.add(current)
                
                # Check all adjacent positions
                for dx, dy in self.directions:
                    neighbor = (current[0] + dx, current[1] + dy)
                    if neighbor in positions_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Add the component to the list
            components.append(list(component))
            
            # Remove the positions in this component from the unvisited set
            positions_set -= component
        
        # Sort components by size (largest first)
        components.sort(key=len, reverse=True)
        return components
    
    def select_blocks_for_components(self):
        """
        Select blocks from start state for each disconnected component in goal state
        Blocks are assigned to the component whose centroid they are closest to
        """
        # Calculate distances from each start position to each component centroid
        block_distances = []
        for pos in self.start_positions:
            distances = []
            for i, centroid in enumerate(self.goal_component_centroids):
                # Manhattan distance to component centroid
                dist = abs(pos[0] - centroid[0]) + abs(pos[1] - centroid[1])
                distances.append((dist, i, pos))  # (distance, component_index, position)
            
            # Add the best match for this position
            distances.sort()  # Sort by distance
            block_distances.append(distances[0])  # Keep best match for each block
        
        # Sort all blocks by distance to their closest component
        block_distances.sort()  # This sorts by distance across all blocks
        
        # Initialize component assignments
        component_blocks = {i: [] for i in range(len(self.goal_components))}
        block_component_assignment = {}
        
        # Assign blocks to components based on size requirements
        assigned_blocks = []
        
        # First pass: assign blocks to closest component if there's room
        for dist, comp_idx, pos in block_distances:
            # Skip if this block is already assigned
            if pos in assigned_blocks:
                continue
                
            component = self.goal_components[comp_idx]
            
            # If this component still needs blocks, assign this block to it
            if len(component_blocks[comp_idx]) < len(component):
                component_blocks[comp_idx].append(pos)
                block_component_assignment[pos] = comp_idx
                assigned_blocks.append(pos)
        
        # Second pass: If any components still need blocks, assign remaining blocks
        for comp_idx, blocks in component_blocks.items():
            component = self.goal_components[comp_idx]
            if len(blocks) < len(component):
                # How many more blocks needed for this component
                blocks_needed = len(component) - len(blocks)
                
                # Find closest unassigned blocks
                for dist, _, pos in block_distances:
                    if pos not in assigned_blocks:
                        component_blocks[comp_idx].append(pos)
                        block_component_assignment[pos] = comp_idx
                        assigned_blocks.append(pos)
                        blocks_needed -= 1
                        
                        if blocks_needed == 0:
                            break
        
        # Store the component assignment for later use
        self.block_component_assignment = block_component_assignment
        
        # Return all blocks selected for the goal
        selected_blocks = []
        for blocks in component_blocks.values():
            selected_blocks.extend(blocks)
        
        return selected_blocks
        
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
        
        # For single component, simple 1:1 mapping
        self.block_component_assignment = {pos: 0 for pos in selected_blocks}
        
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
    
    def check_component_connectivity(self, state):
        """
        Check if blocks for each component stay connected to each other
        This is used for multi-component goals
        """
        if not self.multi_component_goal or not self.allow_disconnection:
            return self.is_connected(state)
        
        # Extract target blocks
        target_blocks = [pos for pos in state if pos not in self.non_target_state]
        
        # Group blocks by their assigned component
        component_blocks = {}
        for pos in target_blocks:
            if pos in self.block_component_assignment:
                comp_idx = self.block_component_assignment[pos]
                if comp_idx not in component_blocks:
                    component_blocks[comp_idx] = []
                component_blocks[comp_idx].append(pos)
        
        # Check connectivity within each component
        for comp_blocks in component_blocks.values():
            if len(comp_blocks) > 1 and not self.is_connected(comp_blocks):
                return False
        
        return True
    
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
    
    def get_component_articulation_points(self, state):
        """
        Find articulation points for each component separately
        Used with multi-component goals
        """
        if not self.multi_component_goal or not self.allow_disconnection:
            return self.get_articulation_points(state)
            
        # Extract target blocks
        target_blocks = [pos for pos in state if pos not in self.non_target_state]
        
        # Group blocks by their assigned component
        component_blocks = {}
        for pos in target_blocks:
            if pos in self.block_component_assignment:
                comp_idx = self.block_component_assignment[pos]
                if comp_idx not in component_blocks:
                    component_blocks[comp_idx] = []
                component_blocks[comp_idx].append(pos)
        
        # Get articulation points for each component
        component_articulation_points = set()
        for comp_blocks in component_blocks.values():
            if len(comp_blocks) > 1:
                articulation_points = self.get_articulation_points(set(comp_blocks))
                component_articulation_points.update(articulation_points)
        
        return component_articulation_points
    
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
                
                # For connectivity check, adapt based on goal structure
                if self.allow_disconnection:
                    if self.multi_component_goal:
                        # Check connectivity within each component
                        if self.check_component_connectivity(new_state):
                            valid_moves.append(new_state)
                    else:
                        # For single component with disconnection allowed
                        target_blocks = [pos for pos in new_state if pos not in self.non_target_state]
                        if self.is_connected(target_blocks) or len(target_blocks) <= 1:
                            valid_moves.append(new_state)
                else:
                    # Standard connectivity check for regular goals
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
            
            if self.multi_component_goal:
                # For multi-component goals, get articulation points within each component
                articulation_points = self.get_component_articulation_points(state_set)
                movable_points = movable_candidates - articulation_points
                
                # If all points in a component are critical, try moving one anyway
                if not movable_points and articulation_points:
                    # Group blocks by component
                    component_blocks = {}
                    for pos in movable_candidates:
                        if pos in self.block_component_assignment:
                            comp_idx = self.block_component_assignment[pos]
                            if comp_idx not in component_blocks:
                                component_blocks[comp_idx] = []
                            component_blocks[comp_idx].append(pos)
                    
                    # For each component with all articulation points
                    for comp_idx, blocks in component_blocks.items():
                        comp_art_points = [p for p in blocks if p in articulation_points]
                        if len(comp_art_points) == len(blocks) and len(blocks) > 1:
                            # Pick one articulation point to move
                            for point in comp_art_points:
                                # Try removing and see if component remains connected
                                temp_comp_blocks = blocks.copy()
                                temp_comp_blocks.remove(point)
                                if len(temp_comp_blocks) <= 1 or self.is_connected(temp_comp_blocks):
                                    movable_points.add(point)
                                    break
            else:
                # For single-component goals with disconnection
                target_blocks = list(movable_candidates)
                if target_blocks:
                    target_articulation_points = self.get_articulation_points(set(target_blocks))
                    movable_points = set(target_blocks) - target_articulation_points
                    
                    # If all target points are critical, try moving one anyway
                    if not movable_points and target_articulation_points and len(target_blocks) > 1:
                        for point in target_articulation_points:
                            # Try removing and see if structure remains connected
                            temp_target_blocks = set(target_blocks)
                            temp_target_blocks.remove(point)
                            if self.is_connected(temp_target_blocks) or len(temp_target_blocks) <= 1:
                                movable_points.add(point)
                                break
                else:
                    movable_points = set()
        else:
            # Standard connectivity rules apply to all blocks
            articulation_points = self.get_articulation_points(state_set)
            movable_points = state_set - articulation_points
            
            # If all points are critical, try moving one anyway
            if not movable_points and articulation_points and len(state_set) > 1:
                for point in articulation_points:
                    # Try removing and see if structure remains connected
                    temp_state = state_set.copy()
                    temp_state.remove(point)
                    if self.is_connected(temp_state) or len(temp_state) <= 1:
                        movable_points.add(point)
                        break
        
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
                    # With disconnection allowed, we need to check:
                    # 1. No overlap between target and fixed blocks
                    new_target_blocks = [pos for pos in new_state_set if pos not in self.non_target_state]
                    new_fixed_blocks = [pos for pos in new_state_set if pos in self.non_target_state]
                    
                    # Skip if any target block occupies the same position as a fixed block
                    if any(pos in new_fixed_blocks for pos in new_target_blocks):
                        continue
                    
                    # 2. For multi-component goals, check connectivity within each component
                    if self.multi_component_goal:
                        # Check if connectivity is maintained within each component
                        if self.check_component_connectivity(new_state_set):
                            single_moves.append((point, new_pos))
                    else:
                        # For single component, just check target block connectivity
                        if self.is_connected(new_target_blocks) or len(new_target_blocks) <= 1:
                            single_moves.append((point, new_pos))
                else:
                    # Standard connectivity check for regular goals
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
                            
                        # For multi-component goals, check connectivity within each component
                        if self.multi_component_goal:
                            if self.check_component_connectivity(new_state):
                                valid_moves.append(frozenset(new_state))
                        else:
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
                    
                    # For multi-component goals, check connectivity within each component
                    if self.multi_component_goal:
                        if self.check_component_connectivity(new_state):
                            valid_moves.append(frozenset(new_state))
                    else:
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
                
            # Determine goal position based on component assignment
            closest_goal = None
            min_dist = float('inf')
            
            if self.multi_component_goal and pos in self.block_component_assignment:
                # Get the component this block is assigned to
                comp_idx = self.block_component_assignment[pos]
                comp_positions = self.goal_components[comp_idx]
                
                # Find closest unoccupied position in this component
                for goal_pos in comp_positions:
                    if goal_pos not in state_set:  # Only consider unoccupied goals
                        dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            closest_goal = goal_pos
            else:
                # Standard closest goal search
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
                        # Extract target and fixed blocks
                        new_target_blocks = [p for p in new_state_set if p not in self.non_target_state]
                        new_fixed_blocks = [p for p in new_state_set if p in self.non_target_state]
                        
                        # Skip if any target block occupies the same position as a fixed block
                        if any(p in new_fixed_blocks for p in new_target_blocks):
                            continue
                            
                        # For multi-component goals, check connectivity within each component
                        if self.multi_component_goal:
                            if self.check_component_connectivity(new_state_set):
                                valid_moves.append(frozenset(new_state_set))
                        else:
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
                    # Extract target and fixed blocks
                    new_target_blocks = [p for p in new_state_set if p not in self.non_target_state]
                    new_fixed_blocks = [p for p in new_state_set if p in self.non_target_state]
                    
                    # Skip if any target block occupies the same position as a fixed block
                    if any(p in new_fixed_blocks for p in new_target_blocks):
                        continue
                        
                    # For multi-component goals, check connectivity within each component
                    if self.multi_component_goal:
                        if self.check_component_connectivity(new_state_set):
                            valid_moves.append(frozenset(new_state_set))
                    else:
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
                
            # Skip if it's a critical articulation point (unless for multi-component goal)
            if not self.allow_disconnection:
                articulation_points = self.get_articulation_points(state_set)
                if pos in articulation_points and len(articulation_points) <= 20:
                    continue
            elif self.allow_disconnection:
                if self.multi_component_goal:
                    # For multi-component goals, check if it's an articulation point in its component
                    if pos in self.block_component_assignment:
                        comp_idx = self.block_component_assignment[pos]
                        comp_blocks = [p for p in state_set if p not in fixed_blocks and 
                                    p in self.block_component_assignment and
                                    self.block_component_assignment[p] == comp_idx]
                        
                        if len(comp_blocks) > 1:
                            comp_articulation_points = self.get_articulation_points(set(comp_blocks))
                            if pos in comp_articulation_points and len(comp_articulation_points) <= 15:
                                continue
                else:
                    # For single component with disconnection
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
                            # Extract target and fixed blocks
                            new_target_blocks = [p for p in new_state_set if p not in self.non_target_state]
                            new_fixed_blocks = [p for p in new_state_set if p in self.non_target_state]
                            
                            # Skip if any target block occupies the same position as a fixed block
                            if any(p in new_fixed_blocks for p in new_target_blocks):
                                continue
                                
                            # For multi-component goals, check connectivity within each component
                            if self.multi_component_goal:
                                if self.check_component_connectivity(new_state_set):
                                    valid_moves.append(frozenset(new_state_set))
                            else:
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
        For multi-component goals, sum distances to appropriate component centroids
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
                
            if self.multi_component_goal:
                # Calculate distance for each component to its centroid
                total_distance = 0
                
                # Group blocks by their assigned component
                component_blocks = {}
                for pos in target_blocks:
                    if pos in self.block_component_assignment:
                        comp_idx = self.block_component_assignment[pos]
                        if comp_idx not in component_blocks:
                            component_blocks[comp_idx] = []
                        component_blocks[comp_idx].append(pos)
                
                # Calculate distance for each component
                for comp_idx, blocks in component_blocks.items():
                    if blocks:
                        comp_centroid = self.calculate_centroid(blocks)
                        goal_centroid = self.goal_component_centroids[comp_idx]
                        total_distance += (abs(comp_centroid[0] - goal_centroid[0]) + 
                                        abs(comp_centroid[1] - goal_centroid[1]))
                
                return total_distance
            else:
                # Standard centroid calculation for target blocks
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
        Handles multi-component goals by matching within each component
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
                
            if self.multi_component_goal:
                # For multi-component goals, calculate distance for each component separately
                
                # Group blocks and goal positions by component
                component_blocks = {}
                component_goals = {}
                
                for pos in target_blocks:
                    if pos in self.block_component_assignment:
                        comp_idx = self.block_component_assignment[pos]
                        if comp_idx not in component_blocks:
                            component_blocks[comp_idx] = []
                        component_blocks[comp_idx].append(pos)
                
                for i, component in enumerate(self.goal_components):
                    component_goals[i] = component
                
                # Calculate distance for each component
                for comp_idx, blocks in component_blocks.items():
                    goals = component_goals.get(comp_idx, [])
                    
                    # Skip if no blocks or goals for this component
                    if not blocks or not goals:
                        continue
                    
                    # If block count doesn't match goal count for this component
                    if len(blocks) != len(goals):
                        return float('inf')
                    
                    # Build distance matrix for this component
                    distance_matrix = []
                    for pos in blocks:
                        row = []
                        for goal_pos in goals:
                            # Manhattan distance
                            dist = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
                            row.append(dist)
                        distance_matrix.append(row)
                    
                    # Greedy assignment for this component
                    assigned_cols = set()
                    
                    for i in range(len(blocks)):
                        # Find closest unassigned goal position
                        min_dist = float('inf')
                        best_j = -1
                        
                        for j in range(len(goals)):
                            if j not in assigned_cols and distance_matrix[i][j] < min_dist:
                                min_dist = distance_matrix[i][j]
                                best_j = j
                        
                        if best_j != -1:
                            assigned_cols.add(best_j)
                            total_distance += min_dist
                        else:
                            # No assignment possible
                            return float('inf')
                    
                    # Add bonus for blocks already in correct positions
                    matching_positions = sum(1 for pos in blocks if pos in goals)
                    total_distance -= matching_positions * 0.5  # Bonus to encourage matches
            else:
                # Standard assignment for single component goal
                goal_list = list(self.goal_state)
                
                # Build distance matrix for all target blocks
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
                
                total_distance += connectivity_bonus
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
            
            total_distance += connectivity_bonus
            
        return total_distance
    
    def block_movement_phase(self, double time_limit=15):
        """
        Phase 1: Move blocks toward the goal centroid or component centroids
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
            if self.multi_component_goal:
                print(f"Moving {len(self.target_block_list)} target blocks towards {len(self.goal_components)} component centroids")
            else:
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
            
            # For multi-component goals, check if blocks are close enough to their component centroids
            if self.allow_disconnection and self.multi_component_goal:
                # Extract target blocks
                target_blocks = [pos for pos in current if pos not in self.non_target_state]
                
                # Group blocks by their assigned component
                component_blocks = {}
                for pos in target_blocks:
                    if pos in self.block_component_assignment:
                        comp_idx = self.block_component_assignment[pos]
                        if comp_idx not in component_blocks:
                            component_blocks[comp_idx] = []
                        component_blocks[comp_idx].append(pos)
                
                # Check if all components are close enough to their centroids
                all_components_close = True
                
                for comp_idx, blocks in component_blocks.items():
                    if blocks:
                        comp_centroid = self.calculate_centroid(blocks)
                        goal_centroid = self.goal_component_centroids[comp_idx]
                        component_distance = (abs(comp_centroid[0] - goal_centroid[0]) + 
                                           abs(comp_centroid[1] - goal_centroid[1]))
                        
                        # If any component is too far, continue movement
                        if component_distance > max_distance:
                            all_components_close = False
                            break
                
                if all_components_close:
                    print(f"All components reached appropriate distances from their target centroids")
                    return self.reconstruct_path(came_from, current)
            
            # For single-component goals, check distance to centroid as before
            elif self.allow_disconnection:
                # Extract target blocks
                target_blocks = [pos for pos in current if pos not in self.non_target_state]
                
                # Calculate centroid of target blocks only
                if target_blocks:
                    target_centroid = self.calculate_centroid(target_blocks)
                    centroid_distance = (abs(target_centroid[0] - self.goal_centroid[0]) + 
                                        abs(target_centroid[1] - self.goal_centroid[1]))
                else:
                    centroid_distance = float('inf')
                        
                if min_distance <= centroid_distance <= max_distance:
                    print(f"Blocks stopped 1 grid cell before goal centroid. Distance: {centroid_distance}")
                    return self.reconstruct_path(came_from, current)
            else:
                # Standard centroid calculation for all blocks
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
                
                    # Adjusted heuristic based on goal structure
                    if self.allow_disconnection and self.multi_component_goal:
                        # For multi-component goals, use the block_heuristic directly
                        # which handles component-specific distances
                        adjusted_heuristic = self.block_heuristic(neighbor)
                    elif self.allow_disconnection:
                        # For single-component goals with disconnection
                        neighbor_target_blocks = [pos for pos in neighbor if pos not in self.non_target_state]
                        if neighbor_target_blocks:
                            neighbor_centroid = self.calculate_centroid(neighbor_target_blocks)
                            neighbor_distance = (abs(neighbor_centroid[0] - self.goal_centroid[0]) + 
                                               abs(neighbor_centroid[1] - self.goal_centroid[1]))
                            
                            # Penalize distances that are too small
                            distance_penalty = 0
                            if neighbor_distance < min_distance:
                                distance_penalty = 10 * (min_distance - neighbor_distance)
                            
                            adjusted_heuristic = self.block_heuristic(neighbor) + distance_penalty
                        else:
                            adjusted_heuristic = float('inf')
                    else:
                        # Standard centroid calculation for regular goals
                        neighbor_centroid = self.calculate_centroid(neighbor)
                        neighbor_distance = (abs(neighbor_centroid[0] - self.goal_centroid[0]) + 
                                           abs(neighbor_centroid[1] - self.goal_centroid[1]))
                
                        # Penalize distances that are too small
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
                        # Find state with appropriate distance to centroid
            best_state = None
            best_distance_diff = float('inf')
            
            for state in came_from.keys():
                # Skip states with overlapping blocks
                if self.has_overlapping_blocks(state):
                    continue
                
                # Handle multi-component goals
                if self.allow_disconnection and self.multi_component_goal:
                    # Extract target blocks
                    target_blocks = [pos for pos in state if pos not in self.non_target_state]
                    
                    # Group blocks by their assigned component
                    component_blocks = {}
                    for pos in target_blocks:
                        if pos in self.block_component_assignment:
                            comp_idx = self.block_component_assignment[pos]
                            if comp_idx not in component_blocks:
                                component_blocks[comp_idx] = []
                            component_blocks[comp_idx].append(pos)
                    
                    # Calculate total distance difference across all components
                    total_distance_diff = 0
                    all_within_range = True
                    
                    for comp_idx, blocks in component_blocks.items():
                        if blocks:
                            comp_centroid = self.calculate_centroid(blocks)
                            goal_centroid = self.goal_component_centroids[comp_idx]
                            component_distance = (abs(comp_centroid[0] - goal_centroid[0]) + 
                                               abs(comp_centroid[1] - goal_centroid[1]))
                            
                            # Calculate difference from acceptable range
                            if component_distance < min_distance:
                                total_distance_diff += min_distance - component_distance
                                all_within_range = False
                            elif component_distance > max_distance:
                                total_distance_diff += component_distance - max_distance
                                all_within_range = False
                    
                    # If all components are within range, this is the best state
                    if all_within_range:
                        best_state = state
                        break
                    
                    # Otherwise, track the state with minimal deviation
                    if total_distance_diff < best_distance_diff:
                        best_distance_diff = total_distance_diff
                        best_state = state
                
                # For single-component goals or standard goals
                else:
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
                if self.allow_disconnection and self.multi_component_goal:
                    # Extract target blocks
                    target_blocks = [pos for pos in best_state if pos not in self.non_target_state]
                    
                    # Group blocks by their assigned component
                    component_blocks = {}
                    for pos in target_blocks:
                        if pos in self.block_component_assignment:
                            comp_idx = self.block_component_assignment[pos]
                            if comp_idx not in component_blocks:
                                component_blocks[comp_idx] = []
                            component_blocks[comp_idx].append(pos)
                    
                    # Print distance for each component
                    for comp_idx, blocks in component_blocks.items():
                        if blocks:
                            comp_centroid = self.calculate_centroid(blocks)
                            goal_centroid = self.goal_component_centroids[comp_idx]
                            component_distance = (abs(comp_centroid[0] - goal_centroid[0]) + 
                                               abs(comp_centroid[1] - goal_centroid[1]))
                            print(f"Component {comp_idx+1} reached centroid distance: {component_distance:.2f}")
                
                elif self.allow_disconnection:
                    # Calculate distance for target blocks only
                    target_blocks = [pos for pos in best_state if pos not in self.non_target_state]
                    if target_blocks:
                        best_centroid = self.calculate_centroid(target_blocks)
                        best_distance = (abs(best_centroid[0] - self.goal_centroid[0]) + 
                                       abs(best_centroid[1] - self.goal_centroid[1]))
                        print(f"Best block position found with centroid distance: {best_distance}")
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
        Handles multi-component goals by morphing each component separately
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
            if self.multi_component_goal:
                print(f"Morphing {len(self.target_block_list)} target blocks into {len(self.goal_components)} separate components")
            else:
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
                # For multi-component goals, we need to check each component individually
                if self.multi_component_goal:
                    # Extract target blocks from current state (exclude fixed blocks)
                    target_blocks = [pos for pos in current if pos not in self.non_target_state]
                    
                    # Group blocks by their assigned component
                    component_blocks = {}
                    for pos in target_blocks:
                        if pos in self.block_component_assignment:
                            comp_idx = self.block_component_assignment[pos]
                            if comp_idx not in component_blocks:
                                component_blocks[comp_idx] = []
                            component_blocks[comp_idx].append(pos)
                    
                    # Check if each component matches its goal
                    all_components_match = True
                    for comp_idx, blocks in component_blocks.items():
                        goal_component = self.goal_components[comp_idx]
                        
                        # Check if this component matches its goal shape
                        component_frozenset = frozenset(blocks)
                        goal_component_frozenset = frozenset(goal_component)
                        
                        if not (component_frozenset.issubset(goal_component_frozenset) or 
                                goal_component_frozenset.issubset(component_frozenset)):
                            all_components_match = False
                            break
                    
                    if all_components_match:
                        print(f"All goal components matched after {iterations} iterations!")
                        return self.reconstruct_path(came_from, current)
                    
                    # Alternative goal check: count matching positions across all components
                    total_matching = 0
                    for comp_idx, blocks in component_blocks.items():
                        goal_component = self.goal_components[comp_idx]
                        matching_positions = sum(1 for pos in blocks if pos in goal_component)
                        total_matching += matching_positions
                    
                    if total_matching == len(self.goal_positions):
                        print(f"All goal positions matched after {iterations} iterations!")
                        return self.reconstruct_path(came_from, current)
                else:
                    # For single-component goals with disconnection allowed
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
                    
                    # For multi-component goals, check connectivity within each component
                    if self.multi_component_goal and not self.check_component_connectivity(neighbor):
                        continue
                    # For single component, check if target blocks stay connected
                    elif not self.multi_component_goal and not (self.is_connected(target_blocks) or len(target_blocks) <= 1):
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
        For multi-component goals, show each component in a different color
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
        
        # Define colors for different components
        component_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        fixed_color = 'lightgray'
        
        # Show initial state
        ax.clear()
        ax.set_xlim(min_x - 0.5, max_x + 0.5)
        ax.set_ylim(min_y - 0.5, max_y + 0.5)
        ax.grid(True)
        
        # Draw goal positions (as outlines)
        if self.multi_component_goal:
            # Draw each component with a different color
            for i, component in enumerate(self.goal_components):
                color = component_colors[i % len(component_colors)]
                for pos in component:
                    rect = plt.Rectangle((pos[1], pos[0]), 1, 1, fill=False, edgecolor=color, linewidth=2)
                    ax.add_patch(rect)
        else:
            # Standard goal visualization
            for pos in self.goal_positions:
                rect = plt.Rectangle((pos[1], pos[0]), 1, 1, fill=False, edgecolor='green', linewidth=2)
                ax.add_patch(rect)
        
        # Draw current positions
        current_positions = path[0]
        rects = []
        
        # Check for duplicates
        if len(current_positions) != len(set(current_positions)):
            print("WARNING: Initial state has overlapping blocks!")
        
        for pos in current_positions:
            if pos in self.fixed_block_list:
                # Fixed blocks in light gray
                rect = plt.Rectangle((pos[1], pos[0]), 1, 1, facecolor=fixed_color, alpha=0.7)
            elif self.multi_component_goal and pos in self.block_component_assignment:
                # Color based on component assignment
                comp_idx = self.block_component_assignment[pos]
                color = component_colors[comp_idx % len(component_colors)]
                rect = plt.Rectangle((pos[1], pos[0]), 1, 1, facecolor=color, alpha=0.7)
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
                    # Fixed blocks in light gray
                    rect = plt.Rectangle((pos[1], pos[0]), 1, 1, facecolor=fixed_color, alpha=0.7)
                elif self.multi_component_goal and pos in self.block_component_assignment:
                    # Color based on component assignment
                    comp_idx = self.block_component_assignment[pos]
                    color = component_colors[comp_idx % len(component_colors)]
                    rect = plt.Rectangle((pos[1], pos[0]), 1, 1, facecolor=color, alpha=0.7)
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