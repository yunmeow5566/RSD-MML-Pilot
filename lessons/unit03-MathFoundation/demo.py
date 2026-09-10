import random

import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class PlotHelper:
    def _build_line_points(self, m:float, b:float, x_range:tuple=(-10, 10), num_points:int=100):
        x_values = [x_range[0] + i * (x_range[1] - x_range[0]) / (num_points - 1) for i in range(num_points)]
        y_values = [m * x + b for x in x_values]
        return x_values, y_values

    def _extract_x_y_values(self, points:list[Point]):
        point_x_values = [point.x if isinstance(point, Point) else point[0] for point in points]
        point_y_values = [point.y if isinstance(point, Point) else point[1] for point in points]
        return point_x_values, point_y_values

    def overlay_points_on_line(self, points:list[Point], m:float, b:float, x_range:tuple=(-10, 10), num_points:int=100, title:str="Line Plot with Points"):
        """Display a line defined by y = mx + b and overlay many Point objects"""
        line_x, line_y = self._build_line_points(m, b, x_range, num_points)
        figure, axes = plt.subplots()
        axes.plot(line_x, line_y, label=f"y = {m}x + {b}", color='blue')

        point_x, point_y = self._extract_x_y_values(points)
        axes.scatter(point_x, point_y, s=12, alpha=0.7, color='red', label='Points')
    
        axes.set_title(title)
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.grid(True, alpha=0.25)
        axes.legend()
        figure.tight_layout()
        return axes

    def plot_line(self, m:float, b:float, x_range:tuple=(-10, 10), num_points:int=100, title:str="Line Plot"):
        """Display a line defined by y = mx + b over a specified range of x values."""
        line_x, line_y = self._build_line_points(m, b, x_range, num_points)
    
        figure, axes = plt.subplots()
        axes.plot(line_x, line_y, label=f"y = {m}x + {b}")
        axes.set_title(title)
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.grid(True, alpha=0.25)
        axes.legend()
        figure.tight_layout()
        return axes

    def plot_points(self, points:list[Point], title="Sample Points"):
        """Display many Point objects or (x, y) pairs as a scatter plot."""
        point_x, point_y = self._extract_x_y_values(points)
        figure, axes = plt.subplots()
        axes.scatter(point_x, point_y, s=12, alpha=0.7)
        axes.set_title(title)
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.grid(True, alpha=0.25)
        figure.tight_layout()
        return axes


if __name__ == "__main__":
    sample_points = [Point(x-10, 2 * (x-10) + 1 + random.uniform(-1, 1)) for x in range(20)]
    # Add more points for demo purpose
    sample_points.append(Point(0.36-10, 2 * (0.36-10) + 1.5))
    plot_helper = PlotHelper()
    plot_helper.plot_points(sample_points)
    plot_helper.plot_line(m=2, b=1, x_range=(-10, 10), num_points=100, title="Line Plot: y = 2x + 1")
    plot_helper.overlay_points_on_line(points=sample_points, m=1.5, b = 2)
    plt.show()

