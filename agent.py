from typing import List, Tuple
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt

# TODO: Put everything into velocity space?


class Agent:
    """ Simple implementation of a goal-directed agent. No collision avoidance as the 
    velocity is always the normed goal vector scaled by max speed.
    """
    DT = 1.0 / 60.0
    RADIUS = 1
    MAX_SPEED = 1.3

    def __init__(self, pos: Tuple[float, float], goal: Tuple[float, float]) -> None:
        """Creates an agent at a specified position with some goal. All agents have 
        a fixed radius and maximum speed that is defined in the Agent class.

        Args:
            pos (Tuple[float, float]): Starting position
            goal (Tuple[float, float]): Goal position
        """
        super().__init__()
        self._pos = np.array(pos, dtype="float")
        self._goal = np.array(goal, dtype="float")
        self._vel = self._goal_vel
        
    @property
    def _goal_vel(self) -> np.array:
        """Return the goal vector based off of the current position and goal

        Returns:
            np.array: goal velocity
        """
        self._vel = self._goal - self._pos
        return self.clip(self._vel, Agent.MAX_SPEED)

    def clip(self, vec: np.array, limit: float) -> np.array:
        """Clip the magnitude of the array to some limit

        Args:
            vec (np.array): Vector to limit
            limit (float): Limit to use

        Returns:
            np.array: Clipped vector
        """
        if np.linalg.norm(vec) > limit:
            return vec / np.linalg.norm(vec) * limit

        return vec

    def calculateForce(self, neighbors: List["Agent"]) -> np.array:
        """Calculate the "forces" that are induced upon the ego agent
        by its' neighbors. For the Agent class it is just a simple goal
        directed force.

        Args:
            neighbors (List[Agent]): Neighbors that can induce behavior changes

        Returns:
            np.array: Forces induced by neighbors and goal velocity
        """
        return self._goal_vel - self._vel

    def update(self, neighbors: List["Agent"]) -> np.array:
        """Update the position of the agent using the underlying collision avoidance algorithm

        Args:
            neighbors (List[Agent]): Neighbors that can induce behavior changes

        Returns:
            np.array: Resulting velocity
        """
        # Get the acceleration of the agent
        acc = self.calculateForce(neighbors)

        self._vel += acc * Agent.DT
        self._pos += self._vel * Agent.DT

        return self._vel

class PowerlawAgent(Agent):
    KSI = 0.54
    K = 1.5
    T0 = 3
    M = 2
    MAX_F = 5

    def __init__(self, pos: Tuple[float, float], goal: Tuple[float, float]) -> None:
        """Create a Powerlaw based collision avoidance agent

        Args:
            pos (Tuple[float, float]): Initial position
            goal (Tuple[float, float]): Goal position
        """
        super().__init__(pos, goal)

    def calculateForce(self, neighbors: List["Agent"]) -> np.array:
        """Use the Powerlaw collision avoidance algorithm to derive the forces
        that are induced on the ego agent by the goal force as well as the 
        repulsive forces induced by each neighbor

        Args:
            neighbors (List[Agent]): agents that can induce forces on ego

        Returns:
            np.array: resulting force (acceleration)
        """
        # Start with the force being the difference between current and "optimal" velocity
        F = (self._goal_vel - self._vel) / PowerlawAgent.KSI

        # Loop through neighbors
        for n in neighbors:
            # Calculate some values for checking collisions
            distance_sq = np.linalg.norm(n._pos - self._pos) ** 2
            radius_sq = (Agent.RADIUS * 2) ** 2

            # If the neighbor is not the ego agent then continue. If the distance is
            # equal to the minimum distance then do not induce any forces.
            if n is not self and distance_sq != radius_sq:
                # If agents are colliding use the seperation distance
                if distance_sq < radius_sq:
                    radius_sq = np.sqrt(Agent.RADIUS * 2 - np.sqrt(distance_sq))

                # From the C++ impl of the algorithm
                w = n._pos - self._pos
                v = self._vel - n._vel
                a = np.dot(v, v)
                b = np.dot(w, v)
                c = np.dot(w, w) - radius_sq
                discr = b * b - a * c

                if discr > 0 and not np.isclose(a, 0):
                    discr = np.sqrt(discr)
                    t = (b - discr) / a
                
                    # Make sure there is a collision in the future
                    if t > 0.0:
                        # Calculate the contributing force
                        f = (PowerlawAgent.K * np.exp(-t / PowerlawAgent.T0) * (v - (b * v - a * w) / discr) /
                                (a * np.power(t, PowerlawAgent.M)) * (PowerlawAgent.M / t + 1 / PowerlawAgent.T0))
                        F += f

        return F


if __name__ == "__main__":
    # Set up the command line arguments
    parser = ArgumentParser(description="Generate simulation data using the Powerlaw motion planning algorithm.")
    parser.add_argument("--viz", action="store_true", help="Display simple matplotlib visualization")
    args = parser.parse_args()

    # Define the agents positions and goals
    agent_poss = [(2, 1.5), (2, -1.5)]
    agent_goals = [(-2, 1.5), (-2, -1.5)]
    goal = (10, 0) # ego goal. ego position is always 0, 0

    # Define the size of the "canvas"
    low = -5
    high = 5
    num = 20

    # Store the data and set up the header
    data = []
    header = "x, y, z, vx, vy, " + ", ".join([f"agent_{i}_x, agent_{i}_y" for i in range(len(agent_poss))])

    if args.viz:
        fig, ax = plt.subplots()
        ax.set_aspect("equal")

    # Iterate through the 2D linspace
    for (i, j) in enumerate(np.linspace(low, high, num)):
        for (k, l) in enumerate(np.linspace(low, high, num)):
            # Set up all of the agents
            agents = [
                PowerlawAgent(p, g) for p, g in zip(agent_poss, agent_goals)
            ]
            a2 = PowerlawAgent((j, l), goal)

            # Record all forces induced by each individual agent
            forces = []
            for agent in agents:
                forces.append(a2.calculateForce([agent]))

            # Record the overall velocity
            vel = a2.update(agents)

            # Plot the velocities as a vector field so long as they're out of collision range
            if args.viz and min([np.linalg.norm(np.array([j, l]) - p) for p in agent_poss]) > Agent.RADIUS * 2:
                ax.quiver(j, l, *vel, units="xy")

            # Save the generated data
            data.append(
                np.array([j, l, 0, *vel, *np.array(forces).flatten()])
            )

    # Add the agents and ego goal to the plot and show
    if args.viz:
        for p in agent_poss:
            ax.add_patch(plt.Circle(p, Agent.RADIUS, color="r"))
        ax.scatter(*goal, s=320, marker='*', color='gold', zorder=3)
        plt.show()

    # Save the data to a CSV for analysis in paraview
    np.savetxt("data.csv", np.array(data), delimiter=",", fmt="%f", header=header)
