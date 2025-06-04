import random
from board import Board

class BaseNaiveAgent:
    def __init__(self, name):
        self.name = name
        self.board_size = 10  # Assumed board size; adjust if necessary
        self.board = Board(self.board_size)  # Initialize a board for the agent
        self.board.place_ships()  # Randomly place ships on the board
        self.moves = []
        self.pending_directions = []
        self.played_moves = set()  # Track moves that have been played
        self.last_move = None
        self.last_result = None
        self.reset_moves()

    def reset_moves(self):
        # To be implemented by subclasses
        pass

    def get_next_move(self, board):
        # Prioritize pending targeting directions
        while self.pending_directions:
            direction = self.pending_directions[0]
            while direction['moves']:
                move = direction['moves'].pop(0)
                if move not in self.played_moves:
                    self.played_moves.add(move)
                    return move
            # If no moves left in this direction, drop it
            self.pending_directions.pop(0)

        # Otherwise, use the default move list
        while self.moves:
            move = self.moves.pop(0)
            if move not in self.played_moves:
                self.played_moves.add(move)
                return move
        return None

    def add_targeting(self, hit_coord):
        # Add targeting sequences in order: up, down, right, left
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        for dr, dc in directions:
            moves = []
            for k in range(1, 5):
                r = hit_coord[0] + dr * k
                c = hit_coord[1] + dc * k
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    moves.append((r, c))
                else:
                    break
            if moves:
                self.pending_directions.append({'direction': (dr, dc), 'moves': moves})

    def take_turn(self, board):
        move = self.get_next_move(board)
        if move is None:
            return None
        result = board.attack(move)
        self.last_move = move
        self.last_result = result
        return result


class NaiveAgent1(BaseNaiveAgent):
    def reset_moves(self):
        # Attack all squares left-to-right, top-to-bottom
        self.moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size)]

    def take_turn(self, board):
        move = self.get_next_move(board)
        if move is None:
            return None
        result = board.attack(move)
        self.last_move = move
        self.last_result = result
        return result



class NaiveAgent2(BaseNaiveAgent):
    def reset_moves(self):
        # First attack cells in even columns, then the odd columns
        self.moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if c % 2 == 0]
        self.moves += [(r, c) for r in range(self.board_size) for c in range(self.board_size) if c % 2 == 1]

    def take_turn(self, board):
        move = self.get_next_move(board)
        if move is None:
            return None
        result = board.attack(move)
        self.last_move = move
        self.last_result = result
        return result


class NaiveAgent3(BaseNaiveAgent):
    def reset_moves(self):
        # Same ordering as Agent1
        self.moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size)]

    def take_turn(self, board):
        move = self.get_next_move(board)
        if move is None:
            return None
        result = board.attack(move)
        self.last_move = move
        self.last_result = result
        if result == "hit":
            self.add_targeting(move)
        return result


class NaiveAgent4(BaseNaiveAgent):
    def reset_moves(self):
        # Same ordering as Agent2
        self.moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if c % 2 == 0]
        self.moves += [(r, c) for r in range(self.board_size) for c in range(self.board_size) if c % 2 != 0]

    def take_turn(self, board):
        move = self.get_next_move(board)
        if move is None:
            return None
        result = board.attack(move)
        self.last_move = move
        self.last_result = result
        if result == "hit":
            self.add_targeting(move)
        return result


class NaiveAgent5(BaseNaiveAgent):
    def reset_moves(self):
        # Attack cells in a skip pattern: first use columns at interval of 3, then the rest
        primary_cols = [c for c in range(self.board_size) if c % 3 == 0]
        secondary_cols = [c for c in range(self.board_size) if c not in primary_cols]
        self.moves = [(r, c) for r in range(self.board_size) for c in primary_cols]
        self.moves += [(r, c) for r in range(self.board_size) for c in secondary_cols]

    def take_turn(self, board):
        move = self.get_next_move(board)
        if move is None:
            return None
        result = board.attack(move)
        self.last_move = move
        self.last_result = result
        if result == "hit":
            self.add_targeting(move)
        return result


class NaiveAgent6(BaseNaiveAgent):
    def reset_moves(self):
        # Random ordering of all cells
        self.moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size)]
        random.shuffle(self.moves)

    def take_turn(self, board):
        move = self.get_next_move(board)
        if move is None:
            return None
        result = board.attack(move)
        self.last_move = move
        self.last_result = result
        if result == "hit":
            self.add_targeting(move)
        return result

class NaiveAgent7(BaseNaiveAgent):
    def reset_moves(self):
        # Checkerboard pattern: attack cells where (r+c) is even, then odd
        self.moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if (r+c) % 2 == 0]
        self.moves += [(r, c) for r in range(self.board_size) for c in range(self.board_size) if (r+c) % 2 != 0]

    def take_turn(self, board):
        move = self.get_next_move(board)
        if move is None:
            return None
        result = board.attack(move)
        self.last_move = move
        self.last_result = result
        if result == "hit":
            self.add_targeting(move)
        return result


class NaiveAgent8(BaseNaiveAgent):
    def reset_moves(self):
        # Spiral search from the center outward
        center = self.board_size // 2
        coords = []
        for layer in range(self.board_size):
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if abs(r - center) + abs(c - center) == layer:
                        coords.append((r, c))
            if len(coords) >= self.board_size * self.board_size:
                break
        self.moves = coords[:self.board_size * self.board_size]

    def take_turn(self, board):
        move = self.get_next_move(board)
        if move is None:
            return None
        result = board.attack(move)
        self.last_move = move
        self.last_result = result
        if result == "hit":
            self.add_targeting(move)
        return result


class NaiveAgent9(BaseNaiveAgent):
    def reset_moves(self):
        # Diagonal scanning approach
        coords = []
        for d in range(2 * self.board_size - 1):
            for r in range(self.board_size):
                c = d - r
                if 0 <= c < self.board_size:
                    coords.append((r, c))
        self.moves = coords

    def take_turn(self, board):
        move = self.get_next_move(board)
        if move is None:
            return None
        result = board.attack(move)
        self.last_move = move
        self.last_result = result
        if result == "hit":
            self.add_targeting(move)
        return result


class NaiveAgent10(BaseNaiveAgent):
    def reset_moves(self):
        # Combination of random and systematic approach
        self.moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size)]
        random.shuffle(self.moves)

    def take_turn(self, board):
        move = self.get_next_move(board)
        if move is None:
            return None
        result = board.attack(move)
        self.last_move = move
        self.last_result = result
        if result == "hit":
            # Add adjacent cells in random order as targeting moves
            adjacent = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r, c = move[0] + dr, move[1] + dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    adjacent.append((r, c))
            random.shuffle(adjacent)
            self.pending_directions.append({'direction': None, 'moves': adjacent})
        return result

class UltimateBattleshipAgent(BaseNaiveAgent):
    def reset_moves(self):
        self.moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if (r + c) % 2 == 0]
        random.shuffle(self.moves)
        self.secondary_moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if (r + c) % 2 != 0]
        random.shuffle(self.secondary_moves)
        self.hits = []
        self.mode = "hunt"
        self.target_queue = []

    def get_adjacent_cells(self, coord):
        r, c = coord
        candidates = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
        return [(nr, nc) for nr, nc in candidates if 0 <= nr < self.board_size and 0 <= nc < self.board_size]

    def infer_orientation(self):
        if len(self.hits) >= 2:
            if self.hits[0][0] == self.hits[1][0]:
                return 'horizontal'
            elif self.hits[0][1] == self.hits[1][1]:
                return 'vertical'
        return None

    def expand_along_orientation(self):
        orientation = self.infer_orientation()
        if orientation == 'horizontal':
            rows = set([r for r, _ in self.hits])
            r = rows.pop()
            cols = [c for _, c in self.hits]
            min_c, max_c = min(cols), max(cols)
            options = [(r, min_c-1), (r, max_c+1)]
        elif orientation == 'vertical':
            cols = set([c for _, c in self.hits])
            c = cols.pop()
            rows = [r for r, _ in self.hits]
            min_r, max_r = min(rows), max(rows)
            options = [(min_r-1, c), (max_r+1, c)]
        else:
            return []
        return [(r, c) for r, c in options if 0 <= r < self.board_size and 0 <= c < self.board_size and (r,c) not in self.played_moves]

    def get_next_move(self, board):
        if self.mode == "target" and self.hits:
            expand_options = self.expand_along_orientation()
            random.shuffle(expand_options)
            for move in expand_options:
                if move not in self.played_moves:
                    self.played_moves.add(move)
                    return move
            while self.target_queue:
                move = self.target_queue.pop(0)
                if move not in self.played_moves:
                    self.played_moves.add(move)
                    return move
            self.mode = "hunt"
            self.hits = []

        while self.moves:
            move = self.moves.pop(0)
            if move not in self.played_moves:
                self.played_moves.add(move)
                return move

        while self.secondary_moves:
            move = self.secondary_moves.pop(0)
            if move not in self.played_moves:
                self.played_moves.add(move)
                return move

        return None

    def take_turn(self, board):
        move = self.get_next_move(board)
        if move is None:
            return None
        result = board.attack(move)
        self.last_move = move
        self.last_result = result
        if result == "hit":
            self.hits.append(move)
            if self.mode == "hunt":
                self.mode = "target"
                self.target_queue = self.get_adjacent_cells(move)
            else:
                new_targets = self.get_adjacent_cells(move)
                self.target_queue = [m for m in new_targets if m not in self.played_moves] + self.target_queue
        return result

    def select_targeting_move(self):
        if not self.hits:
            return None

        if len(self.hits) >= 2:
            # Try to infer ship orientation
            if self.hits[0][0] == self.hits[1][0]:  # horizontal
                row = self.hits[0][0]
                cols = [c for _, c in self.hits]
                min_c, max_c = min(cols), max(cols)
                options = [(row, min_c-1), (row, max_c+1)]
            elif self.hits[0][1] == self.hits[1][1]:  # vertical
                col = self.hits[0][1]
                rows = [r for r, _ in self.hits]
                min_r, max_r = min(rows), max(rows)
                options = [(min_r-1, col), (max_r+1, col)]
            else:
                options = []
        else:
            r, c = self.hits[0]
            options = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]

        # Only pick valid moves
        options = [m for m in options if m in self.available_moves]

        if options:
            return random.choice(options)
        else:
            return None