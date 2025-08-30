"""Ship class for Battleship web game."""

class Ship:
    def __init__(self, size, coordinates, orientation):
        self.size = size
        self.coordinates = coordinates  # List of (row, col) tuples
        self.orientation = orientation  # "H" or "V"
        self.hits = set()  # Track which coordinates have been hit
        
    def check_hit(self, coord):
        """Register a hit on this ship."""
        if coord in self.coordinates:
            self.hits.add(coord)
            return True
        return False
    
    def is_sunk(self):
        """Check if all coordinates of this ship have been hit."""
        return len(self.hits) == self.size
    
    def to_dict(self):
        """Convert ship to dictionary for JSON serialization."""
        return {
            'size': self.size,
            'coordinates': list(self.coordinates),
            'orientation': self.orientation,
            'hits': list(self.hits),
            'is_sunk': self.is_sunk()
        }
