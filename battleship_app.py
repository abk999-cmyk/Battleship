import tkinter as tk
from tkinter import ttk, messagebox
import time, csv, os
from tqdm import tqdm
from AI_agent import AIPlayer
from AI_agent2 import AIPlayer2
from AI_agent3 import AIAgent3
from AI_testing_agents import UltimateBattleshipAgent, NaiveAgent6, NaiveAgent10

# Constants
CELL_SIZE = 48
BOARD_PADDING = 2
WATER_COLOR = '#1E90FF'
SHIP_COLOR  = '#808080'
HIT_COLOR   = '#FF4C4C'
MISS_COLOR  = '#4C72B0'

# Simulation settings
TOTAL_GAMES  = 10               # 🔧 set how many games to simulate
# ↑ change TOTAL_GAMES here to run a different batch length
VISUAL_FREQ  = 2               # show every N‑th game live
DATA_DIR     = os.path.join(os.path.dirname(__file__), "game_logs")
os.makedirs(DATA_DIR, exist_ok=True)

class BattleshipUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 AI Battleship Simulator 🤖")
        self.root.geometry("1100x600")
        self.root.configure(bg='#2B3A42')
        self.delay = 800  # milliseconds

        # Load image assets if present
        asset_dir = os.path.join(os.path.dirname(__file__), 'assets')
        self.use_images = all(os.path.exists(os.path.join(asset_dir, img)) for img in ('water.png','ship.png','hit.png','miss.png'))
        if self.use_images:
            self.water_img = tk.PhotoImage(master=self.root, file=os.path.join(asset_dir, 'water.png'))
            self.ship_img  = tk.PhotoImage(master=self.root, file=os.path.join(asset_dir, 'ship.png'))
            self.hit_img   = tk.PhotoImage(master=self.root, file=os.path.join(asset_dir, 'hit.png'))
            self.miss_img  = tk.PhotoImage(master=self.root, file=os.path.join(asset_dir, 'miss.png'))
#########################################################################
        # Initialize AI players
        self.ai1 = AIAgent3("AI with claude")
        self.ai2 = AIPlayer2("AI with google")
        self.turn = 1  # 1 -> ai1, 2 -> ai2
        self.start_time = None
        self.move_count = 0

        # Batch statistics
        self.game_idx      = 0
        self.ai1_wins      = 0
        self.ai2_wins      = 0
        self.ai1_moves_sum = 0
        self.ai2_moves_sum = 0

        # Build UI
        self.build_menu()
        self.build_frames()
        self.build_status_bar()
        self.build_boards()
        self.build_controls()

    def build_menu(self):
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)
        game_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Game", menu=game_menu)
        game_menu.add_command(label="Restart", command=self.restart)
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.root.destroy)

    def build_frames(self):
        style = ttk.Style()
        style.configure("Status.TLabel", background='#2B3A42', foreground='white', font=('Arial', 14))
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(pady=20)
        self.bottom_frame = ttk.Frame(self.root)
        self.bottom_frame.pack(fill='x', pady=10)

    def build_status_bar(self):
        self.status = tk.StringVar()
        lbl = ttk.Label(self.bottom_frame, textvariable=self.status, style="Status.TLabel")
        lbl.pack(fill='x')

    def build_boards(self):
        titles = ['AI-ML', 'AI Baseline']
        self.canvases = {}
        self.cell_maps = {}
        for idx, title in enumerate(titles, start=1):
            frame = ttk.LabelFrame(self.main_frame, text=title, padding=10)
            frame.grid(row=0, column=idx-1, padx=20)
            canvas = tk.Canvas(frame,
                               width=self.ai1.board.size*CELL_SIZE,
                               height=self.ai1.board.size*CELL_SIZE,
                               bg='#1E2A38', highlightthickness=0)
            canvas.pack()
            self.canvases[idx] = canvas
            self.cell_maps[idx] = {}
        # Initially reveal both fleets
        self.draw_all(reveal=True)

    def build_controls(self):
        # Start button
        btn = ttk.Button(self.bottom_frame, text="Start Simulation", command=self.start_simulation)
        btn.pack(side='left', padx=10)
        # Speed slider label
        ttk.Label(self.bottom_frame, text="Speed:").pack(side='left', padx=(20,5))
        # Speed slider
        self.speed_var = tk.IntVar(value=self.delay)
        speed_slider = ttk.Scale(
            self.bottom_frame,
            from_=50,     # fastest
            to=1000,      # slowest
            variable=self.speed_var,
            command=lambda v: setattr(self, 'delay', int(float(v)))
        )
        speed_slider.pack(side='left')
        self.update_status("Ready for simulation.")

    def draw_all(self, reveal=False):
        self.draw_board(self.ai1, self.canvases[1], self.cell_maps[1], reveal)
        self.draw_board(self.ai2, self.canvases[2], self.cell_maps[2], reveal)

    def draw_board(self, ai_player, canvas, mapping, reveal=False):
        canvas.delete('all')
        size = ai_player.board.size
        for r in range(size):
            for c in range(size):
                x, y = c*CELL_SIZE, r*CELL_SIZE
                if self.use_images:
                    mapping[(r,c)] = canvas.create_image(x, y, anchor='nw', image=self.water_img)
                    if reveal and (r,c) in ai_player.board.ship_lookup:
                        canvas.create_image(x, y, anchor='nw', image=self.ship_img)
                else:
                    rect = canvas.create_rectangle(x, y, x+CELL_SIZE, y+CELL_SIZE,
                                                   fill=WATER_COLOR, outline='black')
                    if reveal and (r,c) in ai_player.board.ship_lookup:
                        canvas.create_rectangle(x+5, y+5, x+CELL_SIZE-5, y+CELL_SIZE-5,
                                                fill=SHIP_COLOR, outline='')
                    mapping[(r,c)] = rect  # base cell, not tagged as overlay

    def start_simulation(self):
        self.pbar = tqdm(total=TOTAL_GAMES, desc="Batch progress", ncols=80)
        # Begin a fresh batch
        self.game_idx = 0
        self.ai1_wins = self.ai2_wins = 0
        self.ai1_moves_sum = self.ai2_moves_sum = 0
        self.run_next_game()

    def run_next_game(self):
        """Initialise a single game and (optionally) draw boards."""
        if self.game_idx >= TOTAL_GAMES:
            self.show_batch_summary()
            return

        self.game_idx += 1
        #############################################################################
        self.ai1 = AIAgent3("AI with claude")
        self.ai2 = AIPlayer2("AI with google")
        self.turn = 1
        self.move_count = 0
        self.start_time = time.time()

        # redraw only for visualised games
        if self.game_idx % VISUAL_FREQ == 1:
            # Clear previous overlays to avoid stale graphics
            for cvs in self.canvases.values():
                cvs.delete("overlay_*")
            self.draw_all(reveal=True)
            self.update_status(f"Game {self.game_idx}/{TOTAL_GAMES} – visualised")
            self.root.after(self.delay, self.auto_move)
        else:
            # headless fast play
            self.headless_play()

    def headless_play(self):
        """Run a complete game without UI updates."""
        def step():
            attacker, defender = (self.ai1, self.ai2) if self.turn == 1 else (self.ai2, self.ai1)
            attacker.take_turn(defender.board)
            self.move_count += 1
            if defender.board.all_ships_sunk():
                winner = 'AI‑ML' if self.turn == 1 else 'AI‑Baseline'
                self.finish(winner, headless=True)
                return  # game done
            self.turn = 2 if self.turn == 1 else 1
            # schedule next turn immediately (non‑blocking)
            self.root.after(1, step)
        step()

    def auto_move(self):
        # choose attacker/defender
        if self.turn == 1:
            attacker, defender, canvas_id = self.ai1, self.ai2, 2
        else:
            attacker, defender, canvas_id = self.ai2, self.ai1, 1

        # Let the attacking AI handle its own turn logic
        result = attacker.take_turn(defender.board)
        move = attacker.last_move

        self.move_count += 1
        ship_present = move in defender.board.ship_lookup
        self.mark_cell(self.canvases[canvas_id], move, result, ship_present)

        # check for end
        if defender.board.all_ships_sunk():
            winner = 'AI‑ML' if self.turn == 1 else 'AI‑Baseline'
            return self.finish(winner)

        # swap and schedule next
        self.turn = 2 if self.turn == 1 else 1
        self.update_status(f"{'AI‑ML' if self.turn==1 else 'AI‑Baseline'} is thinking...")
        self.root.after(self.delay, self.auto_move)

    def mark_cell(self, canvas, coord, res, ship_present=False):
        """
        Paint an overlay on `canvas` at board coordinate `coord`
        according to the shot `res` ("hit" / "miss" / "sunk").
        We use a per‑cell tag so any previous overlay is removed first.
        """
        r, c = coord
        x, y = c * CELL_SIZE, r * CELL_SIZE
        tag = f"overlay_{r}_{c}"

        # Remove any prior overlay on this square
        canvas.delete(tag)

        if self.use_images:
            img = self.hit_img if res in ('hit', 'sunk') else self.miss_img
            canvas.create_image(x, y, anchor='nw', image=img, tags=tag)
        else:
            color = HIT_COLOR if res in ('hit', 'sunk') else MISS_COLOR
            canvas.create_rectangle(
                x + 5, y + 5, x + CELL_SIZE - 5, y + CELL_SIZE - 5,
                fill=color, outline='', tags=tag
            )

    def update_status(self, text):
        self.status.set(text)

    def finish(self, winner, headless=False):
        # capture duration immediately before popup
        duration = time.time() - self.start_time
        # update batch stats
        if winner == 'AI‑ML':
            self.ai1_wins += 1
            self.ai1_moves_sum += self.move_count // 2
        else:
            self.ai2_wins += 1
            self.ai2_moves_sum += self.move_count // 2
        # log with correct duration
        self.log_metrics(winner, duration)
        if not headless and self.game_idx % VISUAL_FREQ == 1:
            msg = f"🤖 {winner} wins in {self.move_count//2} moves over {duration:.1f}s! 🤖"
            messagebox.showinfo("Game Over", msg)
        self.update_status(f"Last game winner: {winner}")
        if hasattr(self, 'pbar'):
            self.pbar.update(1)
        # schedule next
        self.run_next_game()

    def log_metrics(self, winner, duration):
        fname = os.path.join(DATA_DIR, 'metrics.csv')
        new = not os.path.exists(fname)
        with open(fname, 'a', newline='') as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(['time','winner','moves','duration_sec'])
            w.writerow([time.strftime("%X"), winner, self.move_count, round(duration, 1)])

        # Dump per‑move history
        try:
            moves_path = os.path.join(DATA_DIR, f"game_{self.game_idx:04d}_moves.csv")
            with open(moves_path, 'w', newline='') as mh:
                mw = csv.writer(mh)
                mw.writerow(['turn','player','row','col','result'])
                for t, (ply, row, col, res) in enumerate(self.ai1.move_log + self.ai2.move_log):
                    mw.writerow([t, ply, row, col, res])
        except Exception:
            pass

    def show_batch_summary(self):
        total_played = self.ai1_wins + self.ai2_wins
        avg_moves_ai1 = self.ai1_moves_sum / self.ai1_wins if self.ai1_wins else 0
        avg_moves_ai2 = self.ai2_moves_sum / self.ai2_wins if self.ai2_wins else 0
        summary = (
            f"Batch complete – {total_played} games\n\n"
            f"AI‑ML wins : {self.ai1_wins} (avg moves {avg_moves_ai1:.1f})\n"
            f"AI‑Base wins: {self.ai2_wins} (avg moves {avg_moves_ai2:.1f})"
        )
        messagebox.showinfo("Batch Summary", summary)
        self.update_status("Batch finished. See summary popup.")
        if hasattr(self, 'pbar'):
            self.pbar.close()

    def restart(self):
        # restart the entire simulation
        self.start_simulation()

if __name__ == '__main__':
    root = tk.Tk()
    app = BattleshipUI(root)
    root.mainloop()