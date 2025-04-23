import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class Visualizer:
    def __init__(self, grid_size, path, start_positions, animation_speed=0.05):
        self.grid_size = grid_size
        self.path = path  # Ensure path is passed correctly
        self.start_positions = start_positions
        self.animation_speed = animation_speed  # Time in seconds between animation steps
        self.paused = False  # Track pause state
        self.animation_done = False  # Track if animation is complete
        self.current_step = 0  # Keep track of the animation step
        self.animation_started = False  # Track if animation has started

        self.fig, self.ax = plt.subplots(figsize=(12, 10))  # Larger figure
        self.fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.95)  # Moved up by adjusting top margin

        # Create button (moved to the right side to make room for radio buttons)
        self.button_ax = self.fig.add_axes([0.7, 0.05, 0.2, 0.075])
        self.button = Button(self.button_ax, "Search")
        
        # Initialize text annotation for status messages
        self.text_annotation = self.ax.text(
            self.grid_size[1] / 2, self.grid_size[0] + 0.3, "", 
            ha="center", fontsize=12, fontweight="bold"
        )

        self.draw_grid()  # Ensure the grid is drawn initially

    def draw_grid(self, highlight_initial=True, highlight_goal=False):
        """Draws the grid and optionally highlights positions."""
        self.ax.clear()
        self.ax.set_xticks(np.arange(self.grid_size[1] + 1), minor=False)
        self.ax.set_yticks(np.arange(self.grid_size[0] + 1), minor=False)
        self.ax.grid(which="major", color="black", linestyle='-', linewidth=0.5)
        self.ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, self.grid_size[1])
        self.ax.set_ylim(0, self.grid_size[0])

        if highlight_initial:
            for x, y in self.start_positions:
                self.ax.add_patch(plt.Rectangle((y, x), 1, 1, color='Grey', label="Start"))
    
        self.text_annotation = self.ax.text(
            self.grid_size[1] / 2, self.grid_size[0] + 0.3, 
            self.text_annotation.get_text(), ha="center", fontsize=12, fontweight="bold"
        )
        plt.draw()
        
    def update_text(self, message, color="black"):
        """Updates the status message above the grid dynamically with color."""
        self.text_annotation.set_text(message)
        self.text_annotation.set_color(color)  # Set text color
        plt.draw()  # Force update

    def handle_button_click(self, event):
        """Handles the button click event for start, pause, resume, and restart."""
        if not self.path:
            print("No path found. Make sure path is set before starting animation.")
            return

        if self.animation_done:
            # Restart the animation
            self.animation_done = False
            self.animation_started = False
            self.current_step = 0  # Reset animation step
            self.button.label.set_text("Start")
            self.update_text("Select a shape for search.", color="blue")
        elif not self.animation_started:
            # Start animation for the first time
            self.animation_started = True
            self.button.label.set_text("Pause")
            self.update_text("Animating...", color="black")
            self.animate_path()
        else:
            # Toggle pause state
            self.paused = not self.paused
            if self.paused:
                self.button.label.set_text("Resume")
                self.update_text("Paused", color="orange")
            else:
                self.button.label.set_text("Pause")
                self.update_text("Animating...", color="black")
                self.animate_path()  # Resume immediately

    def animate_path(self):
        """Animates the path step by step."""
        if not self.path:
            self.update_text("No paths found", color="red")
            return

        # Continue from the current step instead of restarting
        while self.current_step < len(self.path):
            if self.paused:
                return  # Stop animation if paused

            # Draw the current step
            self.ax.clear()  # Clear the grid
            self.ax.set_xticks(np.arange(self.grid_size[1] + 1), minor=False)
            self.ax.set_yticks(np.arange(self.grid_size[0] + 1), minor=False)
            self.ax.grid(which="major", color="black", linestyle='-', linewidth=0.5)
            self.ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
            self.ax.set_aspect('equal')
            self.ax.set_xlim(0, self.grid_size[1])
            self.ax.set_ylim(0, self.grid_size[0])
        
            # Restore text annotation
            self.text_annotation = self.ax.text(
                self.grid_size[1] / 2, self.grid_size[0] + 0.3, 
                "Animating...", ha="center", fontsize=12, fontweight="bold"
            )

            # Get current positions in the path
            positions = self.path[self.current_step]
        
            # Check for overlaps
            if len(set(positions)) < len(positions):
                self.update_text("Warning: Node overlap detected!", color="red")
        
            # Draw each position without labels
            for pos in positions:
                x, y = pos
                self.ax.add_patch(plt.Rectangle((y, x), 1, 1, color='grey'))
            # No more text labels
        
            plt.pause(self.animation_speed)  # Use customizable animation speed
            self.current_step += 1  # Move to the next step

        # Animation completed
        if self.current_step >= len(self.path):
            self.animation_done = True
            self.button.label.set_text("Restart")
            self.update_text("Start another search.", color="green")

        plt.draw()

    def highlight_goal_shape(self, goal_positions):
        """Highlight the goal shape with a dim green color"""
        for pos in goal_positions:
            x, y = pos
            # Convert floating point positions to grid coordinates for display
            if isinstance(x, float) and isinstance(y, float):
                rect = plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color='green', alpha=0.3)
            else:
                rect = plt.Rectangle((y, x), 1, 1, color='green', alpha=0.3)
            self.ax.add_patch(rect)
        plt.draw()

    def set_animation_speed(self, speed):
        """Update the animation speed"""
        self.animation_speed = speed