from __future__ import annotations

import math

from typing import List

from src.models import CarPose, Cone, Path2D


class PathPlanning:
    """Student-implemented path planner.

    You are given the car pose and an array of detected cones, each cone with (x, y, color)
    where color is 0 for yellow (right side) and 1 for blue (left side). The goal is to
    generate a sequence of path points that the car should follow.

    Implement ONLY the generatePath function.
    """

    def __init__(self, car_pose: CarPose, cones: List[Cone]):
        self.car_pose = car_pose
        self.cones = cones

    def generatePath(self) -> Path2D:
        """Return a list of path points (x, y) in world frame.

        Requirements and notes:
        - Cones: color==0 (yellow) are on the RIGHT of the track; color==1 (blue) are on the LEFT.
        - You may be given 2, 1, or 0 cones on each side.
        - Use the car pose (x, y, yaw) to seed your path direction if needed.
        - Return a drivable path that stays between left (blue) and right (yellow) cones.
        - The returned path will be visualized by PathTester.

        The path can contain as many points as you like, but it should be between 5-10 meters,
        with a step size <= 0.5. Units are meters.

        Replace the placeholder implementation below with your algorithm.
        """

        path_length = 8.0
        max_step = 0.5
        half_track_width = 1.0

        car = self.car_pose
        cos_yaw = math.cos(car.yaw)
        sin_yaw = math.sin(car.yaw)
        blue = []
        yellow = []

        for cone in self.cones:
            dx = cone.x - car.x
            dy = cone.y - car.y
            forward = dx * cos_yaw + dy * sin_yaw
            sideways = -dx * sin_yaw + dy * cos_yaw

            if forward <= 0:
                continue
            if cone.color == 1:
                blue.append((forward, sideways))
            elif cone.color == 0:
                yellow.append((forward, sideways))

        blue.sort()
        yellow.sort()

        def boundary_y(side, forward):
            if len(side) == 1 or forward <= side[0][0]:
                return side[0][1]

            first = side[-2]
            second = side[-1]
            for i in range(len(side) - 1):
                if forward <= side[i + 1][0]:
                    first = side[i]
                    second = side[i + 1]
                    break

            x1, y1 = first
            x2, y2 = second
            if abs(x2 - x1) < 0.000001:
                return (y1 + y2) / 2

            fraction = (forward - x1) / (x2 - x1)
            return y1 + (y2 - y1) * fraction

        distances = []
        for forward, sideways in blue + yellow:
            if forward not in distances:
                distances.append(forward)
        distances.sort()

        if distances:
            end = max(path_length, distances[-1] + 1.0)
        else:
            end = path_length
        distances.append(end)
        waypoints = [(0.0, 0.0)]

        for forward in distances:
            if blue and yellow:
                blue_y = boundary_y(blue, forward)
                yellow_y = boundary_y(yellow, forward)
                sideways = (blue_y + yellow_y) / 2
            elif blue:
                sideways = boundary_y(blue, forward) - half_track_width
            elif yellow:
                sideways = boundary_y(yellow, forward) + half_track_width
            else:
                # Default: produce a short straight-ahead path from the current pose.
                # delete/replace this with your own algorithm.
                sideways = 0.0
            waypoints.append((forward, sideways))

        path = [(car.x, car.y)]
        remaining = path_length

        for i in range(len(waypoints) - 1):
            x1, y1 = waypoints[i]
            x2, y2 = waypoints[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            distance = min(length, remaining)
            steps = math.ceil(distance / max_step)
            step_distance = distance / steps

            for step in range(1, steps + 1):
                fraction = step_distance * step / length
                forward = x1 + dx * fraction
                sideways = y1 + dy * fraction
                world_x = car.x + forward * cos_yaw - sideways * sin_yaw
                world_y = car.y + forward * sin_yaw + sideways * cos_yaw
                path.append((world_x, world_y))

            remaining -= distance
            if remaining < 0.000001:
                break

        return path
