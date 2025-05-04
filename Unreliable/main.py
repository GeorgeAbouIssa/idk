from AI_agent import AI_Agent
from Visualizer import Visualizer
import matplotlib.pyplot as plt

# Example usage
grid_size = (10, 10)
start_positions = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)]
goal_positions =  [(8,3),(8,4),(8,5),(8,6),(7,3),(7,6)]
agent = AI_Agent(grid_size, start_positions, goal_positions, topology="moore")
plt.show()
path = agent.search()
vis = Visualizer(grid_size, path, start_positions)
vis.draw_grid()
plt.show()

if path:
    print("Path found! Visualizing...")
else:
    print("No path found.")
    