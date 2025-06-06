from flask import Flask, render_template, request, jsonify, redirect, url_for
from pathlib import Path
import json

import sys
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from battleship_dashboard import GameEngine, AnalyticsManager, AGENT_CLASSES

app = Flask(__name__)

# Global engine and analytics objects
engine = GameEngine()
analytics = AnalyticsManager()

# Helper to serialise board for front-end

def board_to_matrix(board, reveal=False):
    size = board.size
    matrix = []
    for r in range(size):
        row = []
        for c in range(size):
            val = board.grid[r, c]
            cell = val
            if reveal and (r, c) in board.ship_lookup:
                cell = 3  # ship
            row.append(cell)
        matrix.append(row)
    return matrix

@app.route('/')
def index():
    return redirect(url_for('game'))

@app.route('/game', methods=['GET', 'POST'])
def game():
    if request.method == 'POST':
        p1 = request.form.get('player1', 'Human')
        p2 = request.form.get('player2', 'AI_agent4')
        engine.setup_game(p1, p2)
        return redirect(url_for('play'))
    return render_template('game.html', agents=AGENT_CLASSES.keys())

@app.route('/play')
def play():
    if not engine.player1:
        return redirect(url_for('game'))
    board = board_to_matrix(engine.opponent.board, reveal=False)
    return render_template('play.html', board=board)

@app.route('/move', methods=['POST'])
def move():
    data = request.get_json(force=True)
    r = int(data['row'])
    c = int(data['col'])
    result = engine.make_move(r, c)
    board_current = board_to_matrix(engine.opponent.board, reveal=False)
    over = engine.game_over
    winner = engine.winner.name if over else None
    return jsonify({'result': result, 'board': board_current, 'game_over': over, 'winner': winner})

@app.route('/ai_move')
def ai_move():
    if engine.game_over:
        return jsonify({'game_over': True})
    res = engine.ai_move()
    board_current = board_to_matrix(engine.opponent.board, reveal=False)
    over = engine.game_over
    winner = engine.winner.name if over else None
    return jsonify({'result': res, 'board': board_current, 'game_over': over, 'winner': winner})

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'POST':
        p1 = request.form.get('player1')
        p2 = request.form.get('player2')
        n = int(request.form.get('games', 10))
        stats = []
        def progress(current, total, batch_stats, finished=False):
            stats.append(batch_stats.copy())
        engine.run_batch_simulation(p1, p2, n, progress)
        while engine.simulation_thread and engine.simulation_thread.is_alive():
            pass
        summary = stats[-1] if stats else engine.batch_stats
        return render_template('batch.html', agents=AGENT_CLASSES.keys(), summary=summary)
    return render_template('batch.html', agents=AGENT_CLASSES.keys())

@app.route('/analytics')
def analytics_view():
    summary = analytics.get_performance_summary()
    if not summary:
        summary = {'total_games': 0, 'player_stats': {}, 'avg_moves': {}}
    return render_template('analytics.html', summary=json.dumps(summary))

@app.route('/settings')
def settings():
    return render_template('settings.html')

if __name__ == '__main__':
    app.run(debug=True)
