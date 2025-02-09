# must be improved to deal with symmetry on x axis

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

def sample_brain_coordinates(coordinates, num_samples):
    """
    Generates sample points within the bounding box of brain coordinates and returns
    the unique x, y, z values from the points inside the brain using the convex hull approach.

    Parameters:
    - coordinates (np.ndarray): An (N, 3) array of brain coordinates.
    - num_samples (list or tuple): A list of three integers [n_x, n_y, n_z] specifying
      the number of samples along the x, y, and z axes.

    Returns:
    - x_unique (np.ndarray): Unique x-values from the inside points.
    - y_unique (np.ndarray): Unique y-values from the inside points.
    - z_unique (np.ndarray): Unique z-values from the inside points.
    """

    # Ensure that num_samples has three elements
    if len(num_samples) != 3:
        raise ValueError("num_samples must be a list or tuple with three integers [n_x, n_y, n_z]")

    n_x, n_y, n_z = num_samples

    # Compute the convex hull of the brain coordinates
    hull = ConvexHull(coordinates)

    # Get the min and max values for each axis
    x_min, y_min, z_min = coordinates.min(axis=0)
    x_max, y_max, z_max = coordinates.max(axis=0)

    # Handle cases where n_x, n_y, or n_z is 0 or 1
    def generate_axis_samples(n, min_val, max_val):
        if n <= 1:
            # If n is 0 or 1, return the midpoint
            return np.array([(min_val + max_val) / 2])
        else:
            return np.linspace(min_val, max_val, n)

    # Generate sample points along each axis
    x_samples = generate_axis_samples(n_x, x_min, x_max)
    y_samples = generate_axis_samples(n_y, y_min, y_max)
    z_samples = generate_axis_samples(n_z, z_min, z_max)

    # Create a meshgrid of the sample points
    X, Y, Z = np.meshgrid(x_samples, y_samples, z_samples, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()

    # Combine into a single array of sample points
    sample_points = np.vstack((X_flat, Y_flat, Z_flat)).T

    # Function to check if points are inside the convex hull
    def in_hull(points, hull):
        """
        Test if points in `points` are inside the convex hull `hull`.
        """
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull.points[hull.vertices])
        return hull.find_simplex(points) >= 0

    # Check which sample points are inside the convex hull
    inside = in_hull(sample_points, hull)

    # Separate the inside points
    inside_points = sample_points[inside]

    # Extract unique x, y, z values from the inside points
    x_unique = np.unique(inside_points[:, 0])
    y_unique = np.unique(inside_points[:, 1])
    z_unique = np.unique(inside_points[:, 2])

    return x_unique, y_unique, z_unique
