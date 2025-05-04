import sys
import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib.widgets import Button, RadioButtons
from Search_Agent import Search_Agent
from Visualizer import Visualizer

class SearchController:
    def __init__(self, grid_size, formations, topology="moore", time_limit=200):
        self.grid_size = grid_size
        self.formations = formations  # Dictionary of shape names and their goal positions
        self.start_positions = formations["start"]
        self.current_shape = "Ring"  # Default shape
        self.goal_positions = formations["Ring"]
        self.topology = topology
        self.time_limit = time_limit
        self.search_completed = False

        # Enable interactive mode so the grid appears first
        plt.ion()  

        # Create visualizer and show the grid first
        self.vis = Visualizer(grid_size, [], self.start_positions)
        self.vis.draw_grid()
        plt.show(block=False)  # Show grid before printing anything

        # Add radio buttons for shape selection
        self.radio_ax = self.vis.fig.add_axes([0.05, 0.05, 0.25, 0.15])
        self.radio = RadioButtons(
            self.radio_ax, 
            labels=list(formations.keys())[1:],  # Skip "start" key
            active=0  # Default to first option (ring)
        )
        self.radio.on_clicked(self.on_shape_selected)

        # Attach the window close event to stop execution
        self.vis.fig.canvas.mpl_connect('close_event', self.on_close)
        
        self.selection_mode = False  # False = normal mode, True = goal selection mode
        self.custom_goal = []  # Store the custom goal positions
        
        # Add a button to toggle selection mode
        self.select_button_ax = self.vis.fig.add_axes([0.4, 0.05, 0.2, 0.075])
        self.select_button = Button(self.select_button_ax, "Select Goal")
        self.select_button.on_clicked(self.toggle_selection_mode)
        
        # Connect the mouse click event for grid selection
        self.vis.fig.canvas.mpl_connect('button_press_event', self.on_grid_click)

        # Print initialization info
        print(f"Initializing optimized AI Agent...")
        print(f"Grid size: {grid_size}")
        print(f"Start positions: {self.start_positions}")
        print(f"Current shape: {self.current_shape}")
        print(f"Topology: {topology}")
        print(f"Time limit: {time_limit} seconds")

        # Initialize the agent
        self.agent = Search_Agent(grid_size, self.start_positions, self.goal_positions, topology)
        
        # Set up button to start search
        self.vis.button.on_clicked(self.handle_button)
        
        # Print initial assignments
        print("\nElement assignments (start -> goal):")
        for start, goal in self.agent.assignments.items():
            print(f"{start} -> {goal}")
        
        # Update status text
        self.vis.update_text("Select a shape for search.", color="blue")
        
    def toggle_selection_mode(self, event):
        """Toggle between normal mode and goal selection mode"""
        self.selection_mode = not self.selection_mode
        
        if self.selection_mode:
            # Entering selection mode
            self.custom_goal = []  # Clear any previous selection
            self.select_button.label.set_text("Confirm Goal")
            self.vis.update_text("Click on grid cells to define goal state", color="blue")
            
            # Display the current positions as a starting point for goal selection
            self.vis.draw_grid(highlight_goal=False)
            for pos in self.start_positions:
                self.custom_goal.append(pos)
                self.highlight_cell(pos, color='green')
        else:
            # Exiting selection mode, confirm the goal
            if len(self.custom_goal) == len(self.start_positions):
                # Valid goal state with the same number of blocks
                self.goal_positions = self.custom_goal
                # Update agent with new goal
                self.agent = Search_Agent(self.grid_size, self.start_positions, self.goal_positions, self.topology)
                self.select_button.label.set_text("Select Goal")
                self.vis.update_text(f"Custom goal set with {len(self.goal_positions)} blocks", color="green")
                
                # Reset search state
                self.search_completed = False
                self.vis.animation_started = False
                self.vis.animation_done = False
                self.vis.current_step = 0
                self.vis.path = None
                
                # Print updated assignments
                print("\nUpdated element assignments (start -> goal):")
                for start, goal in self.agent.assignments.items():
                    print(f"{start} -> {goal}")
            else:
                # Invalid goal state
                self.select_button.label.set_text("Select Goal")
                self.vis.update_text(f"Invalid goal: Need exactly {len(self.start_positions)} blocks", color="red")
    
    def on_grid_click(self, event):
        """Handle grid cell clicks for goal selection"""
        if not self.selection_mode or event.inaxes != self.vis.ax:
            return
        
        # Convert click coordinates to grid cell
        x = int(event.ydata)  # Reversed for grid coordinates
        y = int(event.xdata)
        pos = (x, y)
        
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            if pos in self.custom_goal:
                # Remove this position
                self.custom_goal.remove(pos)
                # Redraw without this cell
                self.vis.draw_grid(highlight_goal=False)
                for cell_pos in self.custom_goal:
                    self.highlight_cell(cell_pos, color='green')
            else:
                # Add this position if we haven't reached the limit
                if len(self.custom_goal) < len(self.start_positions):
                    self.custom_goal.append(pos)
                    self.highlight_cell(pos, color='green')
                else:
                    # Replace the first position (or use another strategy)
                    old_pos = self.custom_goal.pop(0)
                    self.custom_goal.append(pos)
                    # Redraw everything
                    self.vis.draw_grid(highlight_goal=False)
                    for cell_pos in self.custom_goal:
                        self.highlight_cell(cell_pos, color='green')
            
            # Update the counter
            self.vis.update_text(f"Selected {len(self.custom_goal)}/{len(self.start_positions)} blocks", color="blue")
    
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
        self.current_shape = label
        self.goal_positions = self.formations[self.current_shape]
        
        # Update agent with new goal positions
        self.agent = Search_Agent(self.grid_size, self.start_positions, self.goal_positions, self.topology)
        
        print(f"Selected shape: {self.current_shape}")
        print(f"Goal positions: {self.goal_positions}")
        
        # Print updated assignments
        print("\nUpdated element assignments (start -> goal):")
        for start, goal in self.agent.assignments.items():
            print(f"{start} -> {goal}")
        
        # Reset search state
        self.search_completed = False
        self.vis.animation_started = False
        self.vis.animation_done = False
        self.vis.current_step = 0
        self.vis.path = None
        
        # Update visualization
        self.vis.draw_grid()
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
        self.vis.update_text("Searching for a path...", color="red")
        plt.pause(0.1)  # Force update to show "Searching..." before search starts
        
        print("\nSearching for optimal path...")
        start_time = time.time()
        path = self.agent.search(self.time_limit)
        search_time = time.time() - start_time
        
        self.search_completed = True

        if path:
            print(f"Path found with {len(path)-1} moves in {search_time:.2f} seconds")
            print(f"Ready for visualization...")
            self.vis.path = path  # Update path in visualizer
            self.vis.button.label.set_text("Start")  # Set button text to Start
            self.vis.update_text(f"Path found ({len(path)-1} moves)", color="green")
            plt.draw()
        else:
            print(f"No path found after {search_time:.2f} seconds")
            self.vis.button.label.set_text("Search")  # Reset button text
            self.vis.update_text("No path found", color="red")
            plt.draw()

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
    
    controller = SearchController(grid_size, formations, "moore", 30)

    plt.ioff()  # Disable interactive mode
    plt.show()  # Keep window open until manually closed