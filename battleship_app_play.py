import tkinter as tk
from tkinter import ttk, messagebox
import time, csv, os, uuid, pathlib, threading
from pathlib import Path
try:
    import pandas as pd
except ImportError:
    pd = None
from player import HumanPlayer
from AI_agent2 import AIPlayer2

# ---- Global settings --------------------------------------------------
DATA_DIR = Path(__file__).with_suffix('').parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Telemetry logger – captures every move and writes CSV/Parquet
# ----------------------------------------------------------------------
class GameLogger:
    def __init__(self):
        self.game_id = uuid.uuid4().hex[:8]
        self.start_ts = time.time()
        self.rows = []

    def log(self, player: str, row: int, col: int, result: str):
        self.rows.append({
            "game_id": self.game_id,
            "timestamp": round(time.time() - self.start_ts, 3),
            "player": player,
            "row": row,
            "col": col,
            "result": result,
        })

    def finish(self, winner: str, move_count: int, duration: float):
        for row in self.rows:
            row["winner"] = winner
            row["move_count"] = move_count
            row["duration"] = round(duration, 3)

        if not self.rows:
            return None

        csv_path = DATA_DIR / "moves.csv"
        parquet_path = DATA_DIR / "moves.parquet"

        # CSV fallback
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.rows[0].keys())
            if new:
                writer.writeheader()
            writer.writerows(self.rows)

        if pd is not None:
            import pandas as _pd
            df = _pd.DataFrame(self.rows)
            try:
                df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
            except Exception:
                pass
            return df
        return None

# Cell dimensions and colors for fallback
CELL_SIZE = 48
BOARD_PADDING = 2
WATER_COLOR = '#1E90FF'
SHIP_COLOR  = '#808080'
HIT_COLOR   = '#FF4C4C'
MISS_COLOR  = '#4C72B0'

class BattleshipUI:
    def __init__(self, root):
        self.root = root
        self.root.title("⚓ Battleship Showdown ⚓")
        self.root.geometry("1100x600")
        self.root.configure(bg='#2B3A42')
        self.move_count = 0
        # Match‑level score keeping
        self.player_score = 0
        self.ai_score = 0

        # Determine if image assets exist
        asset_dir = os.path.join(os.path.dirname(__file__), 'assets')
        self.use_images = all(os.path.exists(os.path.join(asset_dir, img)) for img in ['water.png','ship.png','hit.png','miss.png'])
        if self.use_images:
            self.water_img = tk.PhotoImage(master=self.root, file=os.path.join(asset_dir, 'water.png'))
            self.ship_img  = tk.PhotoImage(master=self.root, file=os.path.join(asset_dir, 'ship.png'))
            self.hit_img   = tk.PhotoImage(master=self.root, file=os.path.join(asset_dir, 'hit.png'))
            self.miss_img  = tk.PhotoImage(master=self.root, file=os.path.join(asset_dir, 'miss.png'))

        # Initialize game logic
        self.setup_game()
        # Build UI
        self.build_menu()
        self.build_frames()
        self.build_status_bar()
        self.build_boards()
        self.update_status("Ready to sail! Click a cell on the Enemy Board.")

    def setup_game(self):
        self.logger = GameLogger()
        self.human = HumanPlayer("Player", manual_setup='n')
        self.ai = AIPlayer2("Computer")
        self.move_count = 0
        self.start_time = time.time()
        self.human_cells = {}
        self.ai_cells = {}

    def build_menu(self):
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)
        game_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Game", menu=game_menu)
        game_menu.add_command(label="Restart Game", command=self.restart)
        game_menu.add_command(label="Reset Scores", command=self.reset_scores)
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.root.destroy)

    def build_frames(self):
        style = ttk.Style()
        # Adopt clam theme for a cleaner appearance
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Status.TLabel", background='#2B3A42', foreground='white', font=('Arial', 14))
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(pady=20)
        self.bottom_frame = ttk.Frame(self.root)
        self.bottom_frame.pack(fill='x', pady=10)

    def build_status_bar(self):
        self.status = tk.StringVar()
        self.score_text = tk.StringVar()
        self.score_text.set("Score – You 0 : 0 AI")
        status_lbl = ttk.Label(self.bottom_frame, textvariable=self.status, style="Status.TLabel")
        status_lbl.pack(fill='x')
        score_lbl = ttk.Label(self.bottom_frame, textvariable=self.score_text, style="Status.TLabel")
        score_lbl.pack(fill='x')

        # Control buttons
        btn_frame = ttk.Frame(self.bottom_frame)
        btn_frame.pack(pady=6)
        ttk.Button(btn_frame, text="Next Game", command=self.restart, width=12).pack(side='left', padx=10)
        ttk.Button(btn_frame, text="Reset Scores", command=self.reset_scores, width=12).pack(side='left', padx=10)

    def build_boards(self):
        titles = ['Your Fleet', 'Enemy Waters']
        for idx, title in enumerate(titles):
            frame = ttk.LabelFrame(self.main_frame, text=title, padding=10)
            frame.grid(row=0, column=idx, padx=20)
            canvas = tk.Canvas(frame,
                               width=self.human.board.size*CELL_SIZE,
                               height=self.human.board.size*CELL_SIZE,
                               bg='#1E2A38', highlightthickness=0)
            canvas.pack()
            if idx == 0:
                self.human_canvas = canvas
            else:
                self.ai_canvas = canvas
                canvas.bind('<Button-1>', self.on_enemy_click)
        self.draw_all()

    def draw_all(self):
        self.draw_board(self.human, self.human_canvas, self.human_cells, reveal=True)
        self.draw_board(self.ai,    self.ai_canvas,    self.ai_cells,    reveal=False)

    def draw_board(self, player, canvas, mapping, reveal=False):
        canvas.delete('all')
        size = player.board.size
        for r in range(size):
            for c in range(size):
                x = c*CELL_SIZE
                y = r*CELL_SIZE
                if self.use_images:
                    mapping[(r,c)] = canvas.create_image(x, y, anchor='nw', image=self.water_img)
                    if reveal and (r,c) in player.board.ship_lookup:
                        canvas.create_image(x, y, anchor='nw', image=self.ship_img)
                else:
                    rect = canvas.create_rectangle(x, y, x+CELL_SIZE, y+CELL_SIZE,
                                                   fill=WATER_COLOR, outline='black')
                    if reveal and (r,c) in player.board.ship_lookup:
                        canvas.create_rectangle(x+5, y+5, x+CELL_SIZE-5, y+CELL_SIZE-5,
                                                fill=SHIP_COLOR, outline='')
                    mapping[(r,c)] = rect

    def on_enemy_click(self, event):
        r = event.y // CELL_SIZE
        c = event.x // CELL_SIZE
        if (r,c) not in self.ai_cells: return
        self.fire(r,c)

    def fire(self, r, c):
        res = self.ai.board.attack((r,c))
        self.logger.log("human", r, c, res)
        self.move_count += 1
        self.mark_cell(self.ai_canvas, (r,c), res)
        if self.ai.board.all_ships_sunk(): return self.finish("Player")
        self.update_status("AI is plotting...")
        self.root.after(400, self.ai_turn)

    def ai_turn(self):
        res = self.ai.take_turn(self.human.board)
        ai_r, ai_c = self.ai.last_move
        self.logger.log("ai", ai_r, ai_c, res)
        self.move_count += 1
        for (r,c) in list(self.human_cells):
            val = self.human.board.grid[r, c]
            if val != 0:
                self.mark_cell(self.human_canvas, (r,c), 'hit' if val==2 else 'miss')
        if self.human.board.all_ships_sunk(): return self.finish("Computer")
        self.update_status("Your turn! Aim carefully.")

    def mark_cell(self, canvas, coord, res):
        r,c = coord
        x,y = c*CELL_SIZE, r*CELL_SIZE
        if self.use_images:
            img = self.hit_img if res in ('hit','sunk') else self.miss_img
            canvas.create_image(x, y, anchor='nw', image=img)
        else:
            color = HIT_COLOR if res in ('hit','sunk') else MISS_COLOR
            canvas.create_rectangle(x+5, y+5, x+CELL_SIZE-5, y+CELL_SIZE-5,
                                    fill=color, outline='')

    def update_status(self, text):
        self.status.set(text)

    def finish(self, winner):
        duration = time.time()-self.start_time
        if winner.lower().startswith("player"):
            self.player_score += 1
        else:
            self.ai_score += 1
        self.update_score_label()
        msg = f"🏆 {winner} wins in {self.move_count/2} moves over {duration:.1f}s! 🏆"
        messagebox.showinfo("Game Over", msg)
        df = self.logger.finish(winner, self.move_count, duration)
        if df is not None and hasattr(self.ai, "learn_from_game"):
            threading.Thread(target=self.ai.learn_from_game, args=(df,), daemon=True).start()
        self.log_metrics(winner)

    def log_metrics(self, winner):
        fname = "metrics.csv"
        new = not os.path.exists(fname)
        with open(fname,'a', newline='') as fh:
            writer = csv.writer(fh)
            if new: writer.writerow(['time','winner','moves','duration'])
            writer.writerow([time.strftime("%X"),winner,self.move_count,round(time.time()-self.start_time,1)])

    def update_score_label(self):
        self.score_text.set(f"Score – You {self.player_score} : {self.ai_score} AI")

    def restart(self):
        self.setup_game()
        self.draw_all()
        self.update_status("Game restarted—fight again!")

    def reset_scores(self):
        self.player_score = 0
        self.ai_score = 0
        self.update_score_label()
        self.update_status("Scores reset. Let the battle begin anew!")

if __name__=='__main__':
    root = tk.Tk()
    app = BattleshipUI(root)
    root.mainloop()
