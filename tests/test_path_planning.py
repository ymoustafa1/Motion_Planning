import math
import unittest

from src.models import CarPose, Cone
from src.path_planning import PathPlanning
from src.scenarios import get_scenario_names, make_scenario


class PathPlanningTests(unittest.TestCase):
    def check_path(self, path, car):
        self.assertGreater(len(path), 1)
        self.assertEqual(path[0], (car.x, car.y))
        self.assertTrue(all(math.isfinite(value) for point in path for value in point))
        gaps = [math.dist(first, second) for first, second in zip(path, path[1:])]
        self.assertLessEqual(max(gaps), 0.5 + 1e-9)
        self.assertAlmostEqual(sum(gaps), 8.0)

    def test_empty_cones_follow_heading(self):
        car = CarPose(3.0, -2.0, math.pi / 3)
        path = PathPlanning(car, []).generatePath()
        self.check_path(path, car)
        for x, y in path:
            self.assertAlmostEqual((x - car.x) * math.sin(car.yaw),
                                   (y - car.y) * math.cos(car.yaw))
        self.assertAlmostEqual(path[-1][0], car.x + 8 * math.cos(car.yaw))
        self.assertAlmostEqual(path[-1][1], car.y + 8 * math.sin(car.yaw))

    def test_straight_boundaries_use_midpoint(self):
        car = CarPose(0.0, 2.0, 0.0)
        cones = [Cone(x, y, color) for x in (2.0, 5.0)
                 for y, color in ((3.0, 1), (1.0, 0))]
        path = PathPlanning(car, cones).generatePath()
        self.check_path(path, car)
        self.assertTrue(all(abs(y - 2.0) < 1e-9 for x, y in path))

    def test_three_cones_on_either_side(self):
        car = CarPose(0.0, 0.0, 0.0)
        for color, side in ((1, 1), (0, -1)):
            with self.subTest(color=color):
                cones = [Cone(x, side * y, color)
                         for x, y in ((2.0, 2.0), (4.0, 3.0), (6.0, 2.0))]
                path = PathPlanning(car, cones).generatePath()
                self.check_path(path, car)
                for point in ((2.0, side), (4.0, 2 * side), (6.0, side)):
                    self.assertTrue(any(math.dist(point, actual) < 1e-9 for actual in path))

    def test_rotation_and_translation(self):
        angle = 0.7

        def transform(x, y):
            return (3 + x * math.cos(angle) - y * math.sin(angle),
                    -4 + x * math.sin(angle) + y * math.cos(angle))

        cones = [Cone(2, 2, 1), Cone(5, 3, 1), Cone(3, -1, 0), Cone(6, 0, 0)]
        original = PathPlanning(CarPose(0, 0, 0), cones).generatePath()
        moved_cones = [Cone(*transform(cone.x, cone.y), cone.color) for cone in cones]
        moved = PathPlanning(CarPose(3, -4, angle), moved_cones).generatePath()
        self.assertEqual(len(original), len(moved))
        for point, actual in zip(original, moved):
            self.assertLess(math.dist(transform(*point), actual), 1e-9)

    def test_cone_order_does_not_change_path(self):
        car = CarPose(0, 0, 0)
        cones = [Cone(2, 2, 1), Cone(5, 3, 1), Cone(3, -1, 0), Cone(6, 0, 0)]
        self.assertEqual(PathPlanning(car, cones).generatePath(),
                         PathPlanning(car, list(reversed(cones))).generatePath())

    def test_duplicate_forward_positions(self):
        car = CarPose(0, 0, 0)
        cones = [Cone(2, 2, 1), Cone(2, 3, 1), Cone(2, -1, 0), Cone(2, -2, 0)]
        self.check_path(PathPlanning(car, cones).generatePath(), car)

    def test_behind_only_uses_straight_path(self):
        car = CarPose(0, 0, 0)
        cones = [Cone(-2, 1, 1), Cone(-3, -1, 0)]
        self.assertEqual(PathPlanning(car, cones).generatePath(),
                         PathPlanning(car, []).generatePath())

    def test_all_scenarios(self):
        for name in get_scenario_names():
            with self.subTest(scenario=name):
                cones, car = make_scenario(name)
                self.check_path(PathPlanning(car, cones).generatePath(), car)


if __name__ == "__main__":
    unittest.main()
