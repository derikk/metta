from typing import List, Optional, Tuple, Union

import numpy as np


class Room:
    """Base class for all room types."""

    def __init__(
        self,
        inside_width: int = 0,
        inside_height: int = 0,
        border_thickness: int = 0,
        border_object: str = "wall",
        labels: Optional[List[str]] = None,
    ):
        assert border_thickness >= 0
        assert inside_width >= 0
        assert inside_height >= 0

        self.inside_width = inside_width
        self.inside_height = inside_height
        self.border_thickness = border_thickness
        self.labels = labels or []

        self.height = inside_height + 2 * border_thickness
        self.width = inside_width + 2 * border_thickness

        assert self.width >= 0
        assert self.height >= 0

        # add area size label
        area = self.width * self.height
        if area < 4000:
            self.labels.append("small")
        elif area < 6000:
            self.labels.append("medium")
        else:
            self.labels.append("large")

        # this is a grid of strings that can accommodate 50 unicode chars
        # don't worry -- we will convert this to an efficient representation later!
        self.grid = np.full((self.height, self.width), "", dtype="<U50")

        # fill in the four border regions
        if border_thickness > 0:
            self._add_borders(border_object)

    def _add_borders(self, border_object: str):
        """Add borders around the room."""
        bt = self.border_thickness
        h, w = self.grid.shape

        # Top and bottom borders
        self.grid[:bt, :] = border_object
        self.grid[h - bt :, :] = border_object

        # Left and right borders
        self.grid[:, :bt] = border_object
        self.grid[:, w - bt :] = border_object

    def fill(self, object_type: str = "wall"):
        """Fill the inner room area with the provided object."""
        bt = self.border_thickness

        # Fill the inner area (excluding border)
        self.grid[bt : bt + self.inside_height, bt : bt + self.inside_width] = object_type

        return self

    def put(self, content: Union["Room", str], x: int, y: int):
        """Place content at the specified position of the grid.

        Args:
            content: Either another Room or a string object type
            x: X coordinate (column) where content should be placed
            y: Y coordinate (row) where content should be placed

        Returns:
            self for method chaining
        """

        if isinstance(content, Room):
            # Place the content from another room
            other_grid = content.grid
            other_h, other_w = other_grid.shape

            # Calculate the area to copy
            max_h = min(y + other_h, self.grid.shape[0])
            max_w = min(x + other_w, self.grid.shape[1])

            # Skip if completely out of bounds
            if x >= self.grid.shape[1] or y >= self.grid.shape[0] or max_w <= 0 or max_h <= 0:
                return self

            # Calculate actual area to copy
            start_y = max(0, y)
            start_x = max(0, x)

            # Calculate offsets in the other grid
            y_offset = max(0, start_y - y)
            x_offset = max(0, start_x - x)

            # Calculate dimensions to copy
            copy_h = max_h - start_y
            copy_w = max_w - start_x

            # overwrite cells from other room
            for i in range(copy_h):
                for j in range(copy_w):
                    other_val = other_grid[y_offset + i, x_offset + j]
                    self.grid[start_y + i, start_x + j] = other_val
        else:
            # Place a single object
            if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]:
                self.grid[y, x] = content

        return self

    def random_position(self, force_inside=True, seed=None) -> Tuple[int, int]:
        """Get a random position within the room.

        Args:
            force_inside: If True, position will be inside the inner area (excluding border)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (x, y) coordinates
        """
        rng = np.random.default_rng(seed)

        if force_inside:
            # Get position within inner area (excluding border)
            bt = self.border_thickness
            x = rng.integers(bt, bt + self.inside_width)
            y = rng.integers(bt, bt + self.inside_height)
        else:
            # Get position anywhere in the grid
            x = rng.integers(0, self.width)
            y = rng.integers(0, self.height)

        return (x, y)

    def put_random(self, content: Union["Room", str], count: int = 1, force_inside=True, seed=None):
        """Place content at random valid positions.

        Args:
            content: Either another Room or a string object type
            count: Number of times to place the content
            force_inside: Limit the insertion point to the area inside the border
            seed: Random seed for reproducibility

        Returns:
            self for method chaining
        """
        rng = np.random.default_rng(seed)

        if isinstance(content, Room):
            # Calculate valid placement range for a room
            if force_inside:
                bt = self.border_thickness
                max_x = (bt + self.inside_width) - content.width + 1
                max_y = (bt + self.inside_height) - content.height + 1
                min_x = bt
                min_y = bt
            else:
                max_x = self.width - content.width + 1
                max_y = self.height - content.height + 1
                min_x = 0
                min_y = 0

            if max_x <= min_x or max_y <= min_y:
                return self  # Other room is too big for placement area

            # Place the room count times
            for _ in range(count):
                # Choose random position
                x = rng.integers(min_x, max_x)
                y = rng.integers(min_y, max_y)

                # Place the room
                self.put(content, x, y)
        else:
            # Place a single object count times
            for _ in range(count):
                x, y = self.random_position(force_inside=force_inside, seed=rng)
                self.put(content, x, y)

        return self

    def build(self):
        """Build this room. Override in subclasses to implement room-specific logic."""
        raise NotImplementedError

    def mirror(self, axis: str = "horizontal"):
        """Mirror the room along the specified axis.

        Args:
            axis: Either "horizontal" or "vertical"

        Returns:
            self for method chaining
        """
        if axis.lower() == "horizontal":
            # Mirror horizontally (flip rows)
            self.grid = np.flip(self.grid, axis=0)
        elif axis.lower() == "vertical":
            # Mirror vertically (flip columns)
            self.grid = np.flip(self.grid, axis=1)
        else:
            raise ValueError("Axis must be either 'horizontal' or 'vertical'")

        return self

    def rotate(self, angle: float = 90.0, fill_object: str = ""):
        """Rotate the room by the specified angle in degrees (clockwise).

        Args:
            angle: Rotation angle in degrees, can be any float value
            fill_object: Object to use for filling areas outside the rotated content

        Returns:
            self for method chaining
        """
        # Normalize angle to [0, 360)
        angle = angle % 360

        if angle == 0:
            return self

        # For arbitrary angles, we need to use nearest neighbor sampling
        # Convert angle to radians
        theta = np.radians(angle)

        # Get current dimensions
        h, w = self.grid.shape

        # Save original grid
        original_grid = self.grid.copy()

        # Create a new grid with the same dimensions
        new_grid = np.full((h, w), fill_object, dtype="<U50")

        # Calculate center points
        center_y, center_x = h // 2, w // 2

        # Rotation matrix
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Apply rotation to each point in the new grid
        for y in range(h):
            for x in range(w):
                # Translate to origin
                y_t = y - center_y
                x_t = x - center_x

                # Apply inverse rotation
                src_coords = np.linalg.inv(rot_matrix).dot([y_t, x_t])

                # Translate back and round to nearest integer (nearest neighbor sampling)
                src_y = int(round(src_coords[0] + center_y))
                src_x = int(round(src_coords[1] + center_x))

                # Check if source coordinates are within bounds
                if 0 <= src_y < h and 0 <= src_x < w:
                    new_grid[y, x] = original_grid[src_y, src_x]

        # Update the grid
        self.grid = new_grid

        return self

    def render_on_axis(self, ax, title=None):
        """Render the room on a specific matplotlib axis"""
        import matplotlib.cm as cm
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap

        # Get unique elements in the grid
        unique_elements = set()
        for row in self.grid:
            for cell in row:
                if cell:  # Only add non-empty cells
                    unique_elements.add(cell)

        # Sort elements for consistent coloring
        unique_elements = sorted(list(unique_elements))

        # Create a mapping from elements to integers
        element_to_int = {elem: i for i, elem in enumerate(unique_elements)}

        # Convert grid to numeric representation
        numeric_grid = np.zeros_like(self.grid, dtype=int)
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j]:
                    numeric_grid[i, j] = element_to_int.get(self.grid[i, j], 0)
                else:
                    # Empty cells get a value one higher than all other elements
                    numeric_grid[i, j] = len(unique_elements)

        # Add "empty" to our unique elements list
        unique_elements.append("")

        # Create a colormap with one more color than unique elements
        if len(unique_elements) <= 20:
            # Use tab20 which has 20 distinct colors
            base_cmap = cm.get_cmap("tab20", 20)
            colors = [base_cmap(i % 20) for i in range(len(unique_elements))]
        else:
            # Use hsv which can generate as many colors as needed
            base_cmap = cm.get_cmap("hsv")
            colors = [base_cmap(i / len(unique_elements)) for i in range(len(unique_elements))]

        # Make the last color white for empty cells
        colors[-1] = (1, 1, 1, 1)  # White for empty cells

        # Create custom colormap
        cmap = ListedColormap(colors)

        # Plot the grid
        _im = ax.imshow(numeric_grid, cmap=cmap)

        # Set grid lines
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)

        # Remove major ticks
        ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)

        # Create legend patches
        legend_patches = []
        for i, elem in enumerate(unique_elements):
            # Skip empty cells in the legend
            if i == len(unique_elements) - 1 and elem == "":
                continue
            patch = mpatches.Patch(color=colors[i], label=elem)
            legend_patches.append(patch)

        # Add legend
        if legend_patches:
            ax.legend(handles=legend_patches, loc="center left", bbox_to_anchor=(1, 0.5))

        # Set title if provided
        if title:
            ax.set_title(title)
        else:
            # Use room labels as title if available
            if self.labels:
                ax.set_title(f"Room: {', '.join(self.labels)}")
        ax.set_aspect("equal")

    def copy(self):
        """
        Create a deep copy of this room.

        Returns:
            A new Room instance with the same properties as this one
        """
        # Create a new room with the same basic properties
        new_room = Room(
            inside_width=self.inside_width,
            inside_height=self.inside_height,
            border_thickness=self.border_thickness,
            border_object="",  # We'll copy the grid directly, so no need to add borders
            labels=self.labels.copy() if self.labels else None,
        )

        # Copy the grid
        new_room.grid = np.copy(self.grid)

        # Return the new room
        return new_room


def main():
    """Create and demonstrate room rotation and mirroring with visualization."""
    import matplotlib.pyplot as plt

    # Create a test room
    bt = 3
    room = Room(inside_width=15, inside_height=10, border_thickness=bt, border_object="wall")

    # Fill the room with floor
    room.fill("floor")

    # Add furniture and objects
    table = Room(inside_width=3, inside_height=2, border_thickness=0)
    table.fill("table")
    room.put(table, 5 + bt, 4 + bt)

    # Add three random chairs
    chair = Room(inside_width=2, inside_height=1, border_thickness=0)
    chair.fill("chair")
    room.put_random(chair, 3)

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Original room (top-left)
    room.render_on_axis(axes[0, 0], "Original Room")

    # 45° rotation (top-right)
    rotated_45 = room.copy()
    rotated_45.rotate(45)
    rotated_45.render_on_axis(axes[0, 1], "Rotated 45°")

    # 90° rotation (bottom-left)
    rotated_90 = room.copy()
    rotated_90.rotate(90)
    rotated_90.render_on_axis(axes[1, 0], "Rotated 90°")

    # Mirrored (bottom-right)
    mirrored = room.copy()
    mirrored.mirror(axis="horizontal")
    mirrored.render_on_axis(axes[1, 1], "Mirrored Horizontally")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
