# Powerlaw Explainer

Final project for CPSC 8040 Scientific Visualization. This project aims to explore and explain a common force-based collision avoidance model. For a full explainer you can see my final submission (TODO)

## Brief Explanation
### Goal Directed Movement
Agent's number one task is to get to some predefined goal point. A complex, collision-free motion planner isn't useful if the agent never reaches the goal. One way to enforce this is to have some sort of prefered velocity that drives the agent to the goal. This prefered velocity is usually just a vector pointing to the goal scaled for some maximum speed:

![pref speed equation](https://latex.codecogs.com/png.image?\LARGE&space;\dpi{110}\bg{white}\mathbf{v}_{pref}&space;=&space;\frac{\mathbf{g}&space;-&space;\mathbf{p}}{\lvert&space;\mathbf{g}&space;-&space;\mathbf{p}&space;\rvert}&space;\cdot&space;s_{pref})

where **`g`** is the goal, **`p`** is the current position, and `s` is the maximum agent's speed. We can visualize this as a vector field:

![vector field for goal velocity][./imgs/goal_field.png]