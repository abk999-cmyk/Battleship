class Ship:
    """
    Represents a ship .
    Stores its coordinates, orientation as vertical or horizontal, and tracks hits.
    """

    def __init__(self, size, coordinates, orientation):
        self.size = size
        self.coordinates = coordinates  # (row, col) tuples list
        self.orientation = orientation  # Horizontal or Vertical
        self.hits = set()

    def is_sunk(self):
        # If number of hits on the ship >= size of the ship
        return len(self.hits) >= self.size

    def check_hit(self, move_coord):
        if move_coord in self.coordinates:
            self.hits.add(move_coord)
            return True # Positive hit
        return False # Miss / Negative on hit
