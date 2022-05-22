from math import sin, cos, atan2, radians, degrees, pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
import json

class Robot:

    def __init__(self, body_position, body_heading, head_position, velocity):
        self.body_position = body_position
        self.body_heading = body_heading
        self.head_position = head_position
        self.velocity = velocity
        self.head_side_sway = 0
        self.natural_head_height = head_position[2]
        self.head_rotation = 0

    def move_forward(self, delta_t, t):
        self.body_position += np.array([delta_t*self.velocity*cos(radians(self.body_heading+90)), delta_t*self.velocity*sin(radians(self.body_heading+90))])
        self.move_head_laterally(t)
        self.head_position[0] = self.body_position[0] + self.head_side_sway * cos(radians(self.body_heading))
        self.head_position[1] = self.body_position[1] + self.head_side_sway * sin(radians(self.body_heading))

    def move_head_laterally(self, t):
        self.head_side_sway = 0.1*sin(pi*t)
        theta = atan2(self.head_side_sway, self.natural_head_height)
        self.head_rotation = degrees(theta)
        self.head_position[2] = self.natural_head_height*cos(theta)



ROBOT_PATH = 3
delta_t = 1/20

if ROBOT_PATH == 1:
    robot = Robot(body_position=[-1.6, -3.15], body_heading=0, head_position=[-1.6, -3.15, 0.8], velocity=0.15)
    x = [robot.head_position[0]]
    y = [robot.head_position[1]]

    # Section 1:
    for t in np.arange(delta_t, 11, delta_t):
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

    # Section 2:
    radius = 1.5
    heading_change = degrees(atan2(robot.velocity*delta_t, radius))
    while robot.body_heading > -90:
        t += delta_t
        robot.body_heading -= heading_change
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

elif ROBOT_PATH == 2:
    robot = Robot(body_position=[-2.6, -3.15], body_heading=0, head_position=[-2.6, -3.15, 0.8], velocity=0.15)
    x = [robot.body_position[0]]
    y = [robot.body_position[1]]

    # Section 1:
    radius = 1.5
    heading_change = degrees(atan2(robot.velocity * delta_t, radius))
    target = [-4.5, -0.5]
    relative_heading_to_goal = degrees(atan2(target[1]-robot.body_position[1], target[0]-robot.body_position[0])) - 90 - robot.body_heading
    t = 0
    while relative_heading_to_goal > 0:
        t += delta_t
        robot.body_heading += heading_change
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])
        relative_heading_to_goal = degrees(atan2(target[1]-robot.body_position[1], target[0]-robot.body_position[0])) - 90 - robot.body_heading

    # Section 2:
    while robot.body_position[0] > target[0]:
        t += delta_t
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])
        
    # Section 3:
    radius = 0.25
    heading_change = degrees(atan2(robot.velocity * delta_t, radius))
    while robot.body_heading > -90:
        t += delta_t
        robot.body_heading -= heading_change
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

elif ROBOT_PATH == 3:
    robot = Robot(body_position=[-2.1, -3.15], body_heading=-15, head_position=[-2.1, -3.15, 0.8], velocity=0.15)
    x = [robot.body_position[0]]
    y = [robot.body_position[1]]

    # Section 1:
    radius = 1.5
    heading_change = degrees(atan2(robot.velocity * delta_t, radius))
    t = 0
    while robot.body_heading < 10:
        t += delta_t
        robot.body_heading += heading_change
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

    # Section 2:
    radius = 1
    heading_change = degrees(atan2(robot.velocity * delta_t, radius))
    while robot.body_heading > -90:
        t += delta_t
        robot.body_heading -= heading_change
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

    # Section 3:
    while robot.body_position[0] < 0:
        t += delta_t
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

    # Section 4:
    radius = 1
    heading_change = degrees(atan2(robot.velocity * delta_t, radius))
    while robot.body_heading < 0:
        t += delta_t
        robot.body_heading += heading_change
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

    # Section 5:
    radius = 0.5
    heading_change = degrees(atan2(robot.velocity * delta_t, radius))
    while robot.body_heading > -120:
        t += delta_t
        robot.body_heading -= heading_change
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

    # Section 6:
    while robot.body_position[0] < 3.5:
        t += delta_t
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

    # Section 7:
    radius = 1
    heading_change = degrees(atan2(robot.velocity * delta_t, radius))
    while robot.body_heading < -45:
        t += delta_t
        robot.body_heading += heading_change
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

elif ROBOT_PATH == 4:
    robot = Robot(body_position=[-2.1, 3.15], body_heading=-179, head_position=[-2.1, 3.15, 0.8], velocity=0.15)
    x = [robot.body_position[0]]
    y = [robot.body_position[1]]

    # Section 1:
    radius = 1.5
    heading_change = degrees(atan2(robot.velocity * delta_t, radius))
    t = 0
    while robot.body_heading < -90:
        t += delta_t
        robot.body_heading += heading_change
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

    # Section 2:
    radius = 0.5
    heading_change = degrees(atan2(robot.velocity * delta_t, radius))
    while robot.body_heading > -180:
        t += delta_t
        robot.body_heading -= heading_change
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

    # Section 3:
    radius = 0.5
    robot.body_heading = 180
    heading_change = degrees(atan2(robot.velocity * delta_t, radius))
    while robot.body_heading > 100:
        t += delta_t
        robot.body_heading -= heading_change
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

    # Section 4:
    while robot.body_position[0] > -3:
        t += delta_t
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

    # Section 5:
    radius = 0.85
    heading_change = degrees(atan2(robot.velocity * delta_t, radius))
    while robot.body_heading > 10:
        t += delta_t
        robot.body_heading -= heading_change
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])

    # Section 6:
    while robot.body_position[1] < 2.85:
        t += delta_t
        robot.move_forward(delta_t=delta_t, t=t)
        x.append(robot.head_position[0])
        y.append(robot.head_position[1])



plt.plot(np.asarray(x), np.asarray(y))
plt.axis('scaled')
plt.show()