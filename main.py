import matplotlib
matplotlib.use('TkAgg')  # Try using a more stable backend
import matplotlib.pyplot as plt
import numpy as np
from Visualizer import Visualizer
from ConnectedMatterAgent import ConnectedMatterAgent
import sys
import time

# Debug prints to track execution
print("Starting program...")

# Create grid and initial configuration
grid_size = (10, 10)
print(f"Grid size: {grid_size}")

# Create initial block positions (2x10 line of blocks)
start_positions = []
for i in range(2):
    for j in range(10):
        start_positions.append((i, j))
print(f"Start positions: {start_positions}")

# Define some sample paths for testing
# In your actual code, this would come from your ConnectedMatterAgent
sample_paths = [
    # Each item is a list of positions for blocks at that step
    start_positions,  # Initial positions
    [(i+1, j) for i, j in start_positions],  # Move everything down by 1
    [(i+2, j) for i, j in start_positions]   # Move everything down by 2
]

# Create the visualizer with initial parameters
visualizer = Visualizer(
    grid_size=grid_size,
    path=sample_paths,
    start_positions=start_positions,
    animation_speed=0.5  # Slow down for better visibility
)
print("Visualizer created...")

# Set up button click handler
visualizer.button.on_clicked(visualizer.handle_button_click)

# Update text with instructions
visualizer.update_text("Press 'Search' to start visualization", color="blue")

# Add event handler for window close
def on_close(event):
    print("Window closed by user. Exiting program...")
    sys.exit(0)

# Register the handler
visualizer.fig.canvas.mpl_connect('close_event', on_close)

# Example goal positions (can be replaced with actual goal)
goal_positions = [(5, 5), (5, 6), (6, 5), (6, 6)]  # A small square in the middle
visualizer.highlight_goal_shape(goal_positions)

print("Starting main loop... UI should remain open until manually closed.")

# Use plt.show() in non-blocking mode with a loop to keep the application alive
plt.show(block=False)

try:
    print("UI is now running. Press Ctrl+C in this console to exit.")
    # Keep checking if the figure window is still open
    while plt.fignum_exists(visualizer.fig.number):
        plt.pause(0.1)  # Allow time for UI updates and event processing
        
except KeyboardInterrupt:
    print("Program terminated by user via keyboard interrupt.")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Cleaning up resources...")
    plt.close('all')  # Ensure all matplotlib windows are closed

print("Program has ended.")