import sys
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Button, RadioButtons, TextBox
from ConnectedMatterAgent import ConnectedMatterAgent
from Visualizer import Visualizer
import threading

class SearchController:
    def __init__(self, grid_size, formations, topology="moore", time_limit=1000, max_simultaneous_moves=1, min_simultaneous_moves=1):
        self.grid_size = grid_size
        self.formations = formations  # Dictionary of shape names and their goal positions
        self.start_positions = formations["start"]
        self.current_shape = "Ring"  # Default shape
        self.goal_positions = formations["Ring"]
        self.topology = topology
        self.time_limit = time_limit
        self.max_simultaneous_moves = max_simultaneous_moves
        self.min_simultaneous_moves = min(min_simultaneous_moves, max_simultaneous_moves)
        self.search_completed = False
        self.obstacles = set()  # Store obstacle positions

        # Enable interactive mode so the grid appears first
        plt.ion()  

        # Create visualizer with default animation speed
        self.animation_speed = 0.05  # Default animation speed (seconds)
        self.vis = Visualizer(grid_size, [], self.start_positions, self.animation_speed)
        self.vis.draw_grid()
        plt.show(block=False)  # Show grid before printing anything

        # Add radio buttons for shape selection instead of dropdown
        self.radio_ax = self.vis.fig.add_axes([0.05, 0.05, 0.16, 0.13])
        self.radio = RadioButtons(
            self.radio_ax, 
            labels=list(formations.keys())[1:],  # Skip "start" key
            active=0  # Default to first option (ring)
        )
        self.radio.on_clicked(self.on_shape_selected)

        # Attach the window close event to stop execution
        self.vis.fig.canvas.mpl_connect('close_event', self.on_close)
        
        self.selection_mode = False  # False = normal mode, True = goal selection mode
        self.obstacle_mode = False   # New mode for adding obstacles
        self.custom_goal = []  # Store the custom goal positions
        self.selection_active = False  # New flag to track if selection is currently active
        
        # Add a button to toggle selection mode
        self.select_button_ax = self.vis.fig.add_axes([0.4, 0.05, 0.2, 0.075])
        self.select_button = Button(self.select_button_ax, "Select Goal")
        self.select_button.on_clicked(self.toggle_selection_mode)
        
        # Move obstacle button to the left
        self.obstacle_button_ax = self.vis.fig.add_axes([0.05, 0.34, 0.16, 0.075])
        self.obstacle_button = Button(self.obstacle_button_ax, "Add Obstacles")
        self.obstacle_button.on_clicked(self.toggle_obstacle_mode)
        
        # Add a button to reset obstacles
        self.reset_obstacles_button_ax = self.vis.fig.add_axes([0.05, 0.25, 0.16, 0.075])
        self.reset_obstacles_button = Button(self.reset_obstacles_button_ax, "Reset Obstacles")
        self.reset_obstacles_button.on_clicked(self.reset_obstacles)
        
        # Add label for grid size
        label_ax = self.vis.fig.add_axes([0.82, 0.75, 0.15, 0.05])
        label_ax.text(0.5, 0.5, 'Grid Size', ha='center', va='center')
        label_ax.axis('off')
        
        # Add text input for grid size (centered, narrower)
        self.grid_text_ax = self.vis.fig.add_axes([0.85, 0.7, 0.09, 0.05])  # Narrower width (0.09)
        self.grid_text_box = TextBox(
            self.grid_text_ax, 
            '',  # Remove label since we added it separately above
            initial=f"{grid_size[0]}",
            textalignment='center'  # Center the text in the box
        )
        
        # Add button to apply grid size (centered under text box)
        self.grid_button_ax = self.vis.fig.add_axes([0.845, 0.63, 0.1, 0.05])
        self.grid_button = Button(self.grid_button_ax, "Apply")
        self.grid_button.on_clicked(self.change_grid_size)

        # ==================== MAX SIMULTANEOUS MOVES CONTROL ====================
        # Add background panel for max simultaneous moves
        self.max_moves_panel_ax = self.vis.fig.add_axes([0.81, 0.49, 0.17, 0.1])
        self.max_moves_panel_ax.patch.set_facecolor('lightgray')
        self.max_moves_panel_ax.patch.set_alpha(0.3)
        self.max_moves_panel_ax.axis('off')
        
        # Add label for max simultaneous moves - larger, bolder font
        sim_moves_label_ax = self.vis.fig.add_axes([0.82, 0.55, 0.15, 0.03])
        sim_moves_label_ax.text(0.5, 0.5, 'MAX Simultaneous Moves', 
                               ha='center', va='center', 
                               fontweight='bold', fontsize=9)
        sim_moves_label_ax.axis('off')
        
        # Add slider for max simultaneous moves
        self.sim_moves_slider_ax = self.vis.fig.add_axes([0.84, 0.51, 0.1, 0.03])
        self.sim_moves_slider = plt.Slider(
            self.sim_moves_slider_ax, '',
            1, 5, valinit=self.max_simultaneous_moves, valstep=1
        )
        self.sim_moves_slider.on_changed(self.update_max_simultaneous_moves)
        
        # ==================== MIN SIMULTANEOUS MOVES CONTROL ====================
        # Add background panel for min simultaneous moves
        self.min_moves_panel_ax = self.vis.fig.add_axes([0.81, 0.36, 0.17, 0.1])
        self.min_moves_panel_ax.patch.set_facecolor('lightgray')
        self.min_moves_panel_ax.patch.set_alpha(0.3)
        self.min_moves_panel_ax.axis('off')
        
        # Add label for min simultaneous moves - larger, bolder font
        min_moves_label_ax = self.vis.fig.add_axes([0.82, 0.42, 0.15, 0.03])
        min_moves_label_ax.text(0.5, 0.5, 'MIN Simultaneous Moves', 
                               ha='center', va='center', 
                               fontweight='bold', fontsize=9)
        min_moves_label_ax.axis('off')
        
        # Add slider for min simultaneous moves
        self.min_moves_slider_ax = self.vis.fig.add_axes([0.84, 0.38, 0.1, 0.03])
        self.min_moves_slider = plt.Slider(
            self.min_moves_slider_ax, '',
            1, 5, valinit=self.min_simultaneous_moves, valstep=1
        )
        self.min_moves_slider.on_changed(self.update_min_simultaneous_moves)

        # ==================== ANIMATION SPEED CONTROL ====================
        # Add background panel for animation speed
        self.anim_speed_panel_ax = self.vis.fig.add_axes([0.81, 0.23, 0.17, 0.1])
        self.anim_speed_panel_ax.patch.set_facecolor('lightgray')
        self.anim_speed_panel_ax.patch.set_alpha(0.3)
        self.anim_speed_panel_ax.axis('off')
        
        # Add label for animation speed - larger, bolder font
        anim_speed_label_ax = self.vis.fig.add_axes([0.82, 0.29, 0.15, 0.03])
        anim_speed_label_ax.text(0.5, 0.5, 'Animation Speed', 
                                ha='center', va='center', 
                                fontweight='bold', fontsize=9)
        anim_speed_label_ax.axis('off')
        
        # Add slider for animation speed (in seconds)
        self.anim_speed_slider_ax = self.vis.fig.add_axes([0.84, 0.25, 0.1, 0.03])
        self.anim_speed_slider = plt.Slider(
            self.anim_speed_slider_ax, '',
            0.01, 1.0, valinit=self.animation_speed, valfmt='%.2f'
        )
        self.anim_speed_slider.on_changed(self.update_animation_speed)
        
        # Add a label to explain animation speed
        anim_speed_info_ax = self.vis.fig.add_axes([0.82, 0.21, 0.15, 0.03])
        anim_speed_info_ax.text(0.5, 0.5, 'Seconds per step', 
                               ha='center', va='center', fontsize=8)
        anim_speed_info_ax.axis('off')
        
        
        # Add timer tracking variables to __init__
        self.search_start_time = 0
        self.timer_active = False

        # Add timer display to __init__
        self.timer_ax = self.vis.fig.add_axes([0.05, 0.5, 0.16, 0.075])
        self.timer_ax.axis('off')
        self.timer_text = self.timer_ax.text(0.5, 0.5, "Time: 0.0s", 
                                          ha='center', va='center', 
                                          fontweight='bold', fontsize=15)

        # Print initialization info
        print(f"Initializing Connected Programmable Matter Agent...")
        print(f"Grid size: {grid_size}")
        print(f"Start positions: {self.start_positions}")
        print(f"Current shape: {self.current_shape}")
        print(f"Topology: {topology}")
        print(f"Time limit: {time_limit} seconds")
        print(f"Constraint: All elements must remain connected during movement")
        print(f"Movement capabilities: Block movement, morphing, chain moves, sliding, and snake streaming")

        # Initialize the agent
        self.agent = ConnectedMatterAgent(
            grid_size, 
            self.start_positions, 
            self.goal_positions, 
            topology,
            max_simultaneous_moves=self.max_simultaneous_moves,
            min_simultaneous_moves=self.min_simultaneous_moves,
            obstacles=self.obstacles
        )
        
        # Set up button to start search
        self.vis.button.on_clicked(self.handle_button)
        
        self.dragging = False
        self.drag_start = None
        self.drag_offset = None
        
        # For boundary checking
        self.shape_min_x = 0
        self.shape_max_x = 0
        self.shape_min_y = 0
        self.shape_max_y = 0
        
        # Add mouse events for dragging and selection
        self.vis.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.vis.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.vis.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def toggle_selection_mode(self, event):
        """Toggle between normal mode and goal selection mode"""
        if self.dragging:  # Don't allow mode switch while dragging
            return
        
        # Turn off obstacle mode if it's active
        if self.obstacle_mode:
            self.obstacle_mode = False
            self.obstacle_button.label.set_text("Add Obstacles")
            
        self.selection_mode = not self.selection_mode
        self.selection_active = self.selection_mode
        
        if self.selection_mode:
            # Entering selection mode
            self.custom_goal = []  # Clear any previous selection
            self.select_button.label.set_text("Confirm Goal")
            self.vis.update_text("Click on grid cells to define goal state", color="blue")
            
            # Clear any highlighted goal shape
            self.vis.draw_grid(highlight_goal=False)
            self.vis.highlight_obstacles(self.obstacles)
        else:
            # Exiting selection mode
            if len(self.custom_goal) == len(self.start_positions):
                self.goal_positions = self.custom_goal.copy()  # Make a copy of the custom goal
                self.agent = ConnectedMatterAgent(
                    self.grid_size, 
                    self.start_positions, 
                    self.goal_positions, 
                    self.topology,
                    max_simultaneous_moves=self.max_simultaneous_moves,
                    min_simultaneous_moves=self.min_simultaneous_moves,
                    obstacles=self.obstacles
                )
                self.select_button.label.set_text("Select Goal")
                
                # Update visualization with the new goal shape
                self.vis.draw_grid()
                self.vis.highlight_goal_shape(self.goal_positions)
                self.vis.highlight_obstacles(self.obstacles)
                self.vis.update_text(f"Custom goal set with {len(self.goal_positions)} blocks", color="green")
                
                # Reset search state
                self.search_completed = False
                self.vis.animation_started = False
                self.vis.animation_done = False
                self.vis.current_step = 0
                self.vis.path = None
            else:
                self.selection_mode = True  # Stay in selection mode if invalid
                self.select_button.label.set_text("Select Goal")
                self.vis.update_text(f"Invalid goal: Need exactly {len(self.start_positions)} blocks", color="red")
                
    def toggle_obstacle_mode(self, event):
        """Toggle between normal mode and obstacle placement mode"""
        if self.dragging:  # Don't allow mode switch while dragging
            return
        
        # Turn off selection mode if it's active
        if self.selection_mode:
            self.selection_mode = False
            self.selection_active = False
            self.select_button.label.set_text("Select Goal")
            
        self.obstacle_mode = not self.obstacle_mode
        
        if self.obstacle_mode:
            # Entering obstacle mode
            self.obstacle_button.label.set_text("Confirm Obstacles")
            self.vis.update_text("Click on grid cells to add/remove obstacles", color="orange")
            
            # Redraw grid with existing obstacles
            self.vis.draw_grid()
            self.vis.highlight_goal_shape(self.goal_positions)
            self.vis.highlight_obstacles(self.obstacles)
        else:
            # Exiting obstacle mode
            self.obstacle_button.label.set_text("Add Obstacles")
            
            # Update agent with new obstacles
            self.agent = ConnectedMatterAgent(
                self.grid_size, 
                self.start_positions, 
                self.goal_positions, 
                self.topology,
                max_simultaneous_moves=self.max_simultaneous_moves,
                min_simultaneous_moves=self.min_simultaneous_moves,
                obstacles=self.obstacles
            )
            
            # Update visualization
            self.vis.draw_grid()
            self.vis.highlight_goal_shape(self.goal_positions)
            self.vis.highlight_obstacles(self.obstacles)
            self.vis.update_text(f"{len(self.obstacles)} obstacles placed", color="green")
            
            # Reset search state
            self.search_completed = False
            self.vis.animation_started = False
            self.vis.animation_done = False
            self.vis.current_step = 0
            self.vis.path = None
    
    def reset_obstacles(self, event):
        """Clear all obstacles from the grid"""
        self.obstacles = set()  # Clear the obstacles set
        
        # Exit obstacle mode if it's active
        if self.obstacle_mode:
            self.obstacle_mode = False
            self.obstacle_button.label.set_text("Add Obstacles")
        
        # Update agent with cleared obstacles
        self.agent = ConnectedMatterAgent(
            self.grid_size, 
            self.start_positions, 
            self.goal_positions, 
            self.topology,
            max_simultaneous_moves=self.max_simultaneous_moves,
            min_simultaneous_moves=self.min_simultaneous_moves,
            obstacles=self.obstacles
        )
        
        # Update visualization
        self.vis.draw_grid()
        self.vis.highlight_goal_shape(self.goal_positions)
        self.vis.update_text("All obstacles cleared", color="green")
        
        # Reset search state
        self.search_completed = False
        self.vis.animation_started = False
        self.vis.animation_done = False
        self.vis.current_step = 0
        self.vis.path = None
        self.timer_text.set_text("Time: 0.0s")  # Reset timer display

    def on_grid_click(self, event, x, y):
        """Handle grid cell clicks for goal selection and obstacle placement"""
        pos = (x, y)
        
        # Handle goal selection
        if self.selection_mode and self.selection_active:
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                # Don't allow placing goals on obstacles
                if pos in self.obstacles:
                    return
                
                if pos in self.custom_goal:
                    # Remove this position
                    self.custom_goal.remove(pos)
                else:
                    # Add this position if we haven't reached the limit
                    if len(self.custom_goal) < len(self.start_positions):
                        self.custom_goal.append(pos)
                    else:
                        # Replace the first position
                        self.custom_goal.pop(0)
                        self.custom_goal.append(pos)
                
                # Redraw the grid with current selection
                self.vis.draw_grid(highlight_goal=False)
                for cell_pos in self.custom_goal:
                    self.highlight_cell(cell_pos, color='green')
                self.vis.highlight_obstacles(self.obstacles)
                
                # Update the counter
                self.vis.update_text(f"Selected {len(self.custom_goal)}/{len(self.start_positions)} blocks", color="blue")
        
        # Handle obstacle placement
        elif self.obstacle_mode:
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                # Don't allow placing obstacles on start positions or goal positions
                if pos in self.start_positions or pos in self.goal_positions:
                    return
                    
                if pos in self.obstacles:
                    # Remove this obstacle
                    self.obstacles.remove(pos)
                else:
                    # Add this obstacle
                    self.obstacles.add(pos)
                
                # Redraw the grid with current obstacles
                self.vis.draw_grid()
                self.vis.highlight_goal_shape(self.goal_positions)
                self.vis.highlight_obstacles(self.obstacles)
                
                # Update the counter
                self.vis.update_text(f"{len(self.obstacles)} obstacles placed", color="orange")
                
    def update_timer(self):
        """Update the search timer display"""
        if self.timer_active:
            elapsed = time.time() - self.search_start_time
            self.timer_text.set_text(f"Time: {elapsed:.1f}s")
            plt.pause(0.1)  # Update display
            return True  # Continue timer
        return False  # Stop timer
        
    def highlight_cell(self, pos, color='green'):
        """Highlight a grid cell with the specified color"""
        x, y = pos
        self.vis.ax.add_patch(plt.Rectangle((y, x), 1, 1, color=color, alpha=0.7))
        plt.draw()    

    def on_close(self, event):
        """Stops execution when the Matplotlib window is closed."""
        print("\nWindow closed. Exiting program.")
        sys.exit()  # Forcefully stop the script
        
    def on_shape_selected(self, label):
        """Handle shape selection from radio buttons"""
        # Cancel selection mode if active when changing shapes
        if self.selection_mode:
            self.selection_mode = False
            self.selection_active = False
            self.select_button.label.set_text("Select Goal")
        
        self.current_shape = label
        self.goal_positions = self.formations[self.current_shape]
        
        # Update agent with new goal positions
        self.agent = ConnectedMatterAgent(
            self.grid_size, 
            self.start_positions, 
            self.goal_positions, 
            self.topology,
            max_simultaneous_moves=self.max_simultaneous_moves,
            min_simultaneous_moves=self.min_simultaneous_moves,
            obstacles=self.obstacles
        )
        
        print(f"Selected shape: {self.current_shape}")
        print(f"Goal positions: {self.goal_positions}")
        
        # Reset search state
        self.search_completed = False
        self.vis.animation_started = False
        self.vis.animation_done = False
        self.vis.current_step = 0
        self.vis.path = None
        self.timer_text.set_text("Time: 0.0s")  # Reset timer display
        
        # Update visualization with highlighted goal shape
        self.vis.draw_grid()
        self.vis.highlight_goal_shape(self.goal_positions)
        self.vis.highlight_obstacles(self.obstacles)
        self.vis.button.label.set_text("Search")
        self.vis.update_text(f"Selected {self.current_shape} shape", color="blue")

    def handle_button(self, event):
        """Handle button clicks based on current state"""
        if not self.search_completed:
            self.run_search(event)
        else:
            self.vis.handle_button_click(event)
    
    def run_search(self, event):
        """Runs the search when the Search button is clicked."""
    # Clear any selection or obstacle mode
        self.selection_mode = False
        self.selection_active = False
        self.obstacle_mode = False
        self.select_button.label.set_text("Select Goal")
        self.obstacle_button.label.set_text("Add Obstacles")
    
    # Start the timer
        self.search_start_time = time.time()
        self.timer_text.set_text("Time: 0.0s")
    
    # Clear goal shape highlight
        self.vis.draw_grid()
        self.vis.highlight_obstacles(self.obstacles)
        self.vis.update_text("Searching for a path...", color="red")
        plt.pause(0.1)  # Force update to show "Searching..." before search starts
    
        print("\nSearching for optimal path with connectivity constraint...")
        print(f"Avoiding {len(self.obstacles)} obstacles")
        print(f"Using advanced movement patterns including snake streaming for narrow passages...")
    
    # Run the search
        path = self.agent.search(self.time_limit)
    
    # Calculate elapsed time
        search_time = time.time() - self.search_start_time
    
    # Set final time
        self.timer_text.set_text(f"Time: {search_time:.1f}s")
    
        self.search_completed = True

        if path:
            print(f"Path found with {len(path)-1} moves in {search_time:.2f} seconds")
            print(f"Ready for visualization...")
            self.vis.animation_speed = self.animation_speed  # Ensure animation speed is updated
            self.vis.path = path  # Update path in visualizer
            self.vis.button.label.set_text("Start")  # Set button text to Start
            self.vis.update_text(f"Path found ({len(path)-1} moves)", color="green")
            plt.draw()
        else:
            print(f"No path found after {search_time:.2f} seconds")
            self.vis.button.label.set_text("Search")  # Reset button text
            self.vis.update_text("No paths found", color="red")
            plt.draw()

    def update_max_simultaneous_moves(self, val):
        """Update the maximum number of simultaneous moves"""
        self.max_simultaneous_moves = int(val)
        
        # Ensure min doesn't exceed max
        if self.min_simultaneous_moves > self.max_simultaneous_moves:
            self.min_simultaneous_moves = self.max_simultaneous_moves
            self.min_moves_slider.set_val(self.min_simultaneous_moves)
            
        self.vis.update_text(f"Simultaneous moves: {self.min_simultaneous_moves}-{self.max_simultaneous_moves}", color="blue")
        
        # Update agent with new parameter
        self.agent = ConnectedMatterAgent(
            self.grid_size, 
            self.start_positions, 
            self.goal_positions, 
            self.topology,
            max_simultaneous_moves=self.max_simultaneous_moves,
            min_simultaneous_moves=self.min_simultaneous_moves,
            obstacles=self.obstacles
        )
        
        # Reset search state
        self.search_completed = False
        self.vis.animation_started = False
        self.vis.animation_done = False
        self.vis.current_step = 0
        self.vis.path = None
    
    def update_min_simultaneous_moves(self, val):
        """Update the minimum number of simultaneous moves"""
        self.min_simultaneous_moves = int(val)
        
        # Ensure min doesn't exceed max
        if self.min_simultaneous_moves > self.max_simultaneous_moves:
            self.max_simultaneous_moves = self.min_simultaneous_moves
            self.sim_moves_slider.set_val(self.max_simultaneous_moves)
            
        self.vis.update_text(f"Simultaneous moves: {self.min_simultaneous_moves}-{self.max_simultaneous_moves}", color="blue")
        
        # Update agent with new parameter
        self.agent = ConnectedMatterAgent(
            self.grid_size, 
            self.start_positions, 
            self.goal_positions, 
            self.topology,
            max_simultaneous_moves=self.max_simultaneous_moves,
            min_simultaneous_moves=self.min_simultaneous_moves,
            obstacles=self.obstacles
        )
        
        # Reset search state
        self.search_completed = False
        self.vis.animation_started = False
        self.vis.animation_done = False
        self.vis.current_step = 0
        self.vis.path = None

    def update_animation_speed(self, val):
        """Update the animation speed (seconds between frames)"""
        self.animation_speed = val
        self.vis.set_animation_speed(val)
        self.vis.update_text(f"Animation speed: {val:.2f} sec/step", color="blue")

    def on_text_submit(self, text):
        """Handle grid size text submission"""
        self.change_grid_size(None)  # Call change_grid_size when Enter is pressed
            
    def change_grid_size(self, event):
        """Handle grid size change"""
        try:
            n = int(self.grid_text_box.text)
            if n < 10:
                self.vis.update_text("Grid size must be at least 10", color="red")
                return
            if n > 200:
                self.vis.update_text("Grid size cannot exceed 200", color="red")
                return

            # Update grid size
            self.grid_size = (n, n)
            
            # Clear obstacles when resizing grid
            self.obstacles = set()
            
            # Update text box to show clean number
            self.grid_text_box.set_val(str(n))
            
            # Reinitialize agent with new grid size
            self.agent = ConnectedMatterAgent(
                self.grid_size, 
                self.start_positions, 
                self.goal_positions, 
                self.topology,
                max_simultaneous_moves=self.max_simultaneous_moves,
                min_simultaneous_moves=self.min_simultaneous_moves,
                obstacles=self.obstacles
            )
            
            # Update visualization
            self.vis.grid_size = self.grid_size
            self.vis.draw_grid()
            self.vis.update_text(f"Grid size updated to {n}x{n}", color="green")
            
            # Reset search state
            self.search_completed = False
            self.vis.animation_started = False
            self.vis.animation_done = False
            self.vis.current_step = 0
            self.vis.path = None
            
        except ValueError:
            self.vis.update_text("Invalid grid size. Enter a number between 10-200", color="red")

    def on_mouse_press(self, event):
        """Handle mouse press events for dragging and selection"""
        if event.inaxes != self.vis.ax:
            return
            
        # Convert click coordinates to grid cell
        x, y = int(event.ydata), int(event.xdata)
        
        # Handle selection and obstacle modes separately
        if (self.selection_mode and self.selection_active) or self.obstacle_mode:
            self.on_grid_click(event, x, y)
            return
        
        # Only allow dragging when not in selection or obstacle mode
        click_pos = (x, y)
        
        # Check if click is within the goal shape
        if any(abs(gx - x) < 1 and abs(gy - y) < 1 for gx, gy in self.goal_positions):
            self.dragging = True
            self.drag_start = click_pos
            self.drag_offset = []
            
            # Calculate offsets for all points relative to click position
            for gx, gy in self.goal_positions:
                self.drag_offset.append((gx - x, gy - y))
            
            # Calculate shape boundaries for edge checking
            x_coords = [pos[0] for pos in self.goal_positions]
            y_coords = [pos[1] for pos in self.goal_positions]
            self.shape_min_x = min(x_coords)
            self.shape_max_x = max(x_coords)
            self.shape_min_y = min(y_coords)
            self.shape_max_y = max(y_coords)
            
            # Calculate shape dimensions
            shape_width = self.shape_max_x - self.shape_min_x
            shape_height = self.shape_max_y - self.shape_min_y
            
            # Calculate distance from click to shape bounds
            self.bound_left = x - self.shape_min_x
            self.bound_right = self.shape_max_x - x
            self.bound_top = y - self.shape_min_y
            self.bound_bottom = self.shape_max_y - y
    
    def on_mouse_release(self, event):
        """Handle mouse release events for dragging"""
        if self.dragging and not self.selection_mode:
            self.dragging = False
            if event.inaxes == self.vis.ax:
                # Snap to grid
                x, y = int(event.ydata), int(event.xdata)
                # Apply boundary constraints
                x, y = self.constrain_to_boundaries(x, y)
                
                new_positions = []
                for offset_x, offset_y in self.drag_offset:
                    new_x = x + offset_x
                    new_y = y + offset_y
                    if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
                        new_positions.append((new_x, new_y))
                
                if len(new_positions) == len(self.goal_positions):
                    self.goal_positions = new_positions
                    self.agent = ConnectedMatterAgent(
                        self.grid_size, 
                        self.start_positions, 
                        self.goal_positions, 
                        self.topology,
                        max_simultaneous_moves=self.max_simultaneous_moves,
                        min_simultaneous_moves=self.min_simultaneous_moves,
                        obstacles=self.obstacles
                    )
                    self.search_completed = False
                    self.vis.draw_grid()
                    self.vis.highlight_goal_shape(self.goal_positions)
                    self.vis.highlight_obstacles(self.obstacles)
                    self.vis.update_text("Goal shape moved", color="blue")
    
    def constrain_to_boundaries(self, x, y):
        """Constrain the drag point to keep the shape within grid boundaries"""
        # Constrain x-coordinate
        min_x = self.bound_left  # Minimum allowed x (to keep left edge in bounds)
        max_x = self.grid_size[0] - 1 - self.bound_right  # Maximum allowed x (to keep right edge in bounds)
        x = max(min_x, min(x, max_x))
        
        # Constrain y-coordinate
        min_y = self.bound_top  # Minimum allowed y (to keep top edge in bounds)
        max_y = self.grid_size[1] - 1 - self.bound_bottom  # Maximum allowed y (to keep bottom edge in bounds)
        y = max(min_y, min(y, max_y))
        
        return x, y
    
    def on_mouse_move(self, event):
        """Handle mouse movement events for dragging"""
        if self.dragging and not self.selection_mode and event.inaxes == self.vis.ax:
            # Get cursor position with boundary constraints
            x, y = event.ydata, event.xdata
            x, y = self.constrain_to_boundaries(x, y)
            
            # Clear and redraw the grid
            self.vis.draw_grid()
            
            # Draw the shape at the new position
            temp_positions = []
            for offset_x, offset_y in self.drag_offset:
                new_x = x + offset_x
                new_y = y + offset_y
                temp_positions.append((new_x, new_y))
            
            # Highlight the shape at its temporary position
            self.vis.highlight_goal_shape(temp_positions)

# Example usage
if __name__ == "__main__":
    grid_size = (10, 10)
    
    # Dictionary of formations
    formations = {
        "start": [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (0, 4), (1, 4),
                  (0, 5), (1, 5), (0, 6), (1, 6), (0, 7), (1, 7), (0, 8), (1, 8), (0, 9), (1, 9)],
        
        "Ring": [(7, 4), (7, 5), (6, 3), (6, 4), (6, 5), (6, 6), (5, 2), (5, 3), (5, 6), (5, 7),
                 (4, 2), (4, 3), (4, 6), (4, 7), (3, 3), (3, 4), (3, 5), (3, 6), (2, 4), (2, 5)],
        
        "Rectangle": [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
                      (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7)],
        
        "Triangle": [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 2), (3, 3),
                     (3, 4), (3, 5), (3, 6), (3, 7), (4, 3), (4, 4), (4, 5), (4, 6), (5, 4), (5, 5)]
    }
    
    controller = SearchController(
        grid_size=grid_size, 
        formations=formations, 
        topology="moore", 
        time_limit=1000,
        max_simultaneous_moves=1,  # Default value, can be changed via UI
        min_simultaneous_moves=1   # Default value, can be changed via UI
    )

    plt.ioff()  # Disable interactive mode
    plt.show()  # Keep window open until manually closed