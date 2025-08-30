"""
Battleship AI Research Platform - Web Application
Full-featured web version with advanced AI agents, batch simulation, and analytics
"""

from flask import Flask, render_template, request, jsonify, session, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
import uuid
import json
from datetime import datetime
import threading
import time
import logging
import os
import csv
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import traceback

# Import game logic and AI agents
from game_logic import (
    BattleshipGame, HumanPlayer, AIPlayer,
    AIPlayerClassic, AIPlayer2, AIAgent3, AIAgent4,
    NaiveAgent1, NaiveAgent2, NaiveAgent3, NaiveAgent4, NaiveAgent5,
    NaiveAgent6, NaiveAgent7, NaiveAgent8, NaiveAgent9, NaiveAgent10,
    UltimateBattleshipAgent
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'battleship-ai-research-platform'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure directories exist
Path("logs").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
Path("static/plots").mkdir(exist_ok=True)

# Global storage
games = {}
batch_simulations = {}
performance_metrics = {}

# AI Agent Registry
AI_AGENTS = {
    'basic': AIPlayer,
    'classic': AIPlayerClassic,
    'ai_player2': AIPlayer2,
    'ai_agent3': AIAgent3,
    'ai_agent4': AIAgent4,
    'naive1': NaiveAgent1,
    'naive2': NaiveAgent2,
    'naive3': NaiveAgent3,
    'naive4': NaiveAgent4,
    'naive5': NaiveAgent5,
    'naive6': NaiveAgent6,
    'naive7': NaiveAgent7,
    'naive8': NaiveAgent8,
    'naive9': NaiveAgent9,
    'naive10': NaiveAgent10,
    'ultimate': UltimateBattleshipAgent
}

@app.route('/')
def index():
    """Main game interface."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Advanced analytics dashboard."""
    return render_template('dashboard.html')

@app.route('/batch_simulation')
def batch_simulation():
    """Batch simulation interface."""
    return render_template('batch_simulation.html')

@app.route('/analytics')
def analytics():
    """Performance analytics interface."""
    return render_template('analytics.html')

@app.route('/training')
def training():
    """Model training interface."""
    return render_template('training.html')

# ================================
# Game Management Endpoints
# ================================

@app.route('/api/create_game', methods=['POST'])
def create_game():
    """Create a new game with advanced AI options."""
    try:
        data = request.get_json()
        
        # Extract game parameters
        player1_name = data.get('player1_name', 'Human Player')
        player2_name = data.get('player2_name', 'AI Player')
        game_mode = data.get('game_mode', 'human_vs_ai')
        ai_type = data.get('ai_type', 'ai_agent3')  # Default to best AI
        
        # Create players based on mode and AI type
        if game_mode == 'human_vs_ai':
            player1 = HumanPlayer(player1_name)
            ai_type = data.get('ai_type', 'ai_agent3')
            ai_class = AI_AGENTS.get(ai_type, AIAgent3)
            player2 = ai_class(player2_name)
        elif game_mode == 'ai_vs_ai':
            ai1_type = data.get('ai1_type', 'ai_agent3')
            ai2_type = data.get('ai2_type', 'ultimate')
            ai1_class = AI_AGENTS.get(ai1_type, AIAgent3)
            ai2_class = AI_AGENTS.get(ai2_type, UltimateBattleshipAgent)
            player1 = ai1_class(player1_name)
            player2 = ai2_class(player2_name)
        else:  # human_vs_human
            player1 = HumanPlayer(player1_name)
            player2 = HumanPlayer(player2_name)
        
        # Set player types and auto-place ships for AI players
        if game_mode == 'human_vs_ai':
            player1.player_type = 'human'
            player2.player_type = 'ai'
            # Ships already placed for AI in constructor, but ensure they exist
            if not hasattr(player2.board, 'ships') or not player2.board.ships:
                player2.board.place_ships()
        elif game_mode == 'ai_vs_ai':
            player1.player_type = 'ai'
            player2.player_type = 'ai'
            # Ships already placed for AI in constructor, but ensure they exist
            if not hasattr(player1.board, 'ships') or not player1.board.ships:
                player1.board.place_ships()
            if not hasattr(player2.board, 'ships') or not player2.board.ships:
                player2.board.place_ships()
        else:  # human_vs_human
            player1.player_type = 'human'
            player2.player_type = 'human'
        
        # Create game
        game = BattleshipGame(player1, player2)
        games[game.id] = game
        
        # Store session info
        session['game_id'] = game.id
        session['player_name'] = player1_name
        
        logger.info(f"Game created: {game.id}, Mode: {game_mode}")
        
        # For AI vs AI games, start the auto-play immediately
        if game_mode == 'ai_vs_ai':
            def start_ai_vs_ai_game():
                time.sleep(1)  # Brief delay for UI setup
                try:
                    while not game.is_over():
                        current_ai = game.current_player
                        
                        # Get AI move
                        ai_move = None
                        try:
                            if hasattr(current_ai, 'take_turn'):
                                ai_move = current_ai.take_turn(game.other_player.board)
                            elif hasattr(current_ai, 'attack'):
                                ai_move = current_ai.attack(game.other_player.board)
                            else:
                                # Fallback to random move
                                available_moves = [(r, c) for r in range(10) for c in range(10)
                                                 if game.other_player.board.grid[r, c] == 0]
                                if available_moves:
                                    ai_move = random.choice(available_moves)
                        except Exception as e:
                            logger.error(f"AI move generation error: {str(e)}")
                            # Fallback to random move
                            available_moves = [(r, c) for r in range(10) for c in range(10)
                                             if game.other_player.board.grid[r, c] == 0]
                            if available_moves:
                                ai_move = random.choice(available_moves)
                        
                        if ai_move:
                            move_result = game.make_move(ai_move)
                            
                            # Emit the move to all viewers
                            socketio.emit('ai_move_update', {
                                'game_id': game.id,
                                'player': current_ai.name,
                                'move': ai_move,
                                'result': move_result.get('result', 'miss'),
                                'game_over': move_result.get('game_over', False),
                                'winner': move_result.get('winner'),
                                'game_state': game.to_dict()
                            })
                            
                            logger.info(f"AI move: {current_ai.name} attacks {ai_move} -> {move_result.get('result', 'miss')}")
                            
                            if not game.is_over():
                                time.sleep(1.5)  # Delay between moves for viewing
                        else:
                            break  # No valid moves
                            
                except Exception as e:
                    logger.error(f"AI vs AI game error: {str(e)}")
                    traceback.print_exc()
            
            # Start AI vs AI game in background thread
            threading.Thread(target=start_ai_vs_ai_game).start()
        
        return jsonify({
            'success': True,
            'game_id': game.id,
            'game': game.to_dict(),
            'needs_ship_placement': player1.player_type == 'human' and game_mode != 'ai_vs_ai',
            'is_ai_vs_ai': game_mode == 'ai_vs_ai'
        })
        
    except Exception as e:
        logger.error(f"Error creating game: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/place_ships', methods=['POST'])
def place_ships():
    """Place ships for human player."""
    try:
        data = request.get_json()
        game_id = data.get('game_id')
        player_name = data.get('player_name')
        ship_placements = data.get('ship_placements')
        
        if game_id not in games:
            return jsonify({'error': 'Game not found'}), 404
        
        game = games[game_id]
        
        # Find the player
        if game.player1.name == player_name:
            player = game.player1
        elif game.player2.name == player_name:
            player = game.player2
        else:
            return jsonify({'error': 'Player not found'}), 404
        
        player.place_ships_manual(ship_placements)
        logger.info(f"Ships placed for player {player_name} in game {game_id}")
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error placing ships: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/make_move', methods=['POST'])
def make_move():
    """Make a move in the game."""
    try:
        data = request.get_json()
        game_id = data.get('game_id')
        player_name = data.get('player_name')
        coord = data.get('coord')
        
        if game_id not in games:
            return jsonify({'error': 'Game not found'}), 404
        
        game = games[game_id]
        
        # Verify it's the correct player's turn
        if game.current_player.name != player_name:
            return jsonify({'error': 'Not your turn'}), 400
        
        # Make the move
        result = game.make_move(coord)
        
        if 'error' in result:
            return jsonify(result), 400
        
        # Emit game update to all players
        socketio.emit('game_update', {
            'game_id': game_id,
            'move_result': result,
            'game_state': game.to_dict()
        }, room=game_id)
        
        # If it's now an AI's turn, make AI move
        if not game.is_over() and getattr(game.current_player, 'player_type', 'human') == 'ai':
            def make_ai_move_delayed():
                time.sleep(0.5)  # Small delay for better UX
                try:
                    ai_move = None
                    
                    # Try different AI move methods
                    if hasattr(game.current_player, 'take_turn'):
                        ai_move = game.current_player.take_turn(game.other_player.board)
                    elif hasattr(game.current_player, 'attack'):
                        ai_move = game.current_player.attack(game.other_player.board)
                    else:
                        # Fallback to random move
                        available_moves = [(r, c) for r in range(10) for c in range(10)
                                         if game.other_player.board.grid[r, c] == 0]
                        if available_moves:
                            ai_move = random.choice(available_moves)
                    
                    if ai_move:
                        ai_result = game.make_move(ai_move)
                        socketio.emit('game_update', {
                            'game_id': game_id,
                            'move_result': ai_result,
                            'game_state': game.to_dict()
                        }, room=game_id)
                except Exception as e:
                    logger.error(f"AI move error: {str(e)}")
                    traceback.print_exc()
            
            threading.Thread(target=make_ai_move_delayed).start()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error making move: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ================================
# Batch Simulation Endpoints
# ================================

@app.route('/api/start_batch_simulation', methods=['POST'])
def start_batch_simulation():
    """Start a batch simulation with multiple AI agents."""
    try:
        data = request.get_json()
        simulation_id = str(uuid.uuid4())
        
        # Parse simulation parameters
        num_games = data.get('num_games', 100)
        ai_agents = data.get('ai_agents', ['ai_agent3', 'ultimate'])
        test_against = data.get('test_against', 'all')
        
        # Create simulation task
        simulation = {
            'id': simulation_id,
            'status': 'running',
            'progress': 0,
            'total_games': 0,
            'results': [],
            'start_time': datetime.now(),
            'parameters': data
        }
        
        batch_simulations[simulation_id] = simulation
        
        # Run simulation in background
        def run_simulation():
            try:
                logger.info(f"Starting batch simulation {simulation_id}")
                
                if test_against == 'all':
                    test_agents = list(AI_AGENTS.keys())
                else:
                    test_agents = test_against
                
                total_games = num_games * len(ai_agents) * len(test_agents)
                simulation['total_games'] = total_games
                completed_games = 0
                
                for main_ai_name in ai_agents:
                    main_ai_class = AI_AGENTS[main_ai_name]
                    
                    for opponent_name in test_agents:
                        if main_ai_name == opponent_name:
                            continue  # Skip self-play for now
                        
                        opponent_class = AI_AGENTS[opponent_name]
                        
                        # Run games for this matchup
                        wins = 0
                        losses = 0
                        total_moves = 0
                        game_times = []
                        
                        for game_num in range(num_games):
                            start_time = time.time()
                            
                            # Create players
                            main_ai = main_ai_class(f"{main_ai_name}_main")
                            opponent = opponent_class(f"{opponent_name}_opp")
                            
                            # Auto-place ships
                            main_ai.board.place_ships()
                            opponent.board.place_ships()
                            
                            # Create and run game
                            game = BattleshipGame(main_ai, opponent)
                            winner = game.play()
                            
                            game_time = time.time() - start_time
                            game_times.append(game_time)
                            
                            if winner == main_ai:
                                wins += 1
                            else:
                                losses += 1
                            
                            total_moves += game.move_count
                            completed_games += 1
                            
                            # Update progress
                            simulation['progress'] = int((completed_games / total_games) * 100)
                            
                            # Emit progress update
                            socketio.emit('simulation_progress', {
                                'simulation_id': simulation_id,
                                'progress': simulation['progress'],
                                'completed_games': completed_games,
                                'total_games': total_games
                            })
                        
                        # Store results for this matchup
                        result = {
                            'main_ai': main_ai_name,
                            'opponent': opponent_name,
                            'wins': wins,
                            'losses': losses,
                            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
                            'avg_moves': total_moves / num_games if num_games > 0 else 0,
                            'avg_time': sum(game_times) / len(game_times) if game_times else 0
                        }
                        
                        simulation['results'].append(result)
                        logger.info(f"Completed matchup: {main_ai_name} vs {opponent_name}")
                
                simulation['status'] = 'completed'
                simulation['end_time'] = datetime.now()
                
                # Save results to file
                results_df = pd.DataFrame(simulation['results'])
                results_file = f"data/batch_simulation_{simulation_id}.csv"
                results_df.to_csv(results_file, index=False)
                
                logger.info(f"Batch simulation {simulation_id} completed")
                
                # Emit completion
                socketio.emit('simulation_complete', {
                    'simulation_id': simulation_id,
                    'results': simulation['results']
                })
                
            except Exception as e:
                simulation['status'] = 'error'
                simulation['error'] = str(e)
                logger.error(f"Batch simulation error: {str(e)}")
                socketio.emit('simulation_error', {
                    'simulation_id': simulation_id,
                    'error': str(e)
                })
        
        # Start simulation thread
        threading.Thread(target=run_simulation).start()
        
        return jsonify({
            'success': True,
            'simulation_id': simulation_id
        })
        
    except Exception as e:
        logger.error(f"Error starting batch simulation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_simulation/<simulation_id>')
def get_batch_simulation(simulation_id):
    """Get batch simulation status and results."""
    if simulation_id not in batch_simulations:
        return jsonify({'error': 'Simulation not found'}), 404
    
    simulation = batch_simulations[simulation_id]
    return jsonify(simulation)

@app.route('/api/batch_simulations')
def list_batch_simulations():
    """List all batch simulations."""
    simulations = []
    for sim_id, simulation in batch_simulations.items():
        simulations.append({
            'id': sim_id,
            'status': simulation['status'],
            'progress': simulation['progress'],
            'start_time': simulation['start_time'].isoformat() if simulation['start_time'] else None,
            'total_games': simulation['total_games']
        })
    return jsonify(simulations)

# ================================
# Analytics and Visualization
# ================================

@app.route('/api/generate_performance_plot', methods=['POST'])
def generate_performance_plot():
    """Generate performance visualization plots."""
    try:
        data = request.get_json()
        plot_type = data.get('plot_type', 'win_rates')
        simulation_id = data.get('simulation_id')
        
        if simulation_id not in batch_simulations:
            return jsonify({'error': 'Simulation not found'}), 404
        
        simulation = batch_simulations[simulation_id]
        results_df = pd.DataFrame(simulation['results'])
        
        if results_df.empty:
            return jsonify({'error': 'No results to plot'}), 400
        
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'win_rates':
            # Win rate comparison
            pivot_df = results_df.pivot(index='main_ai', columns='opponent', values='win_rate')
            sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0.5, 
                       fmt='.2f', cbar_kws={'label': 'Win Rate'})
            plt.title('AI Agent Win Rate Matrix')
            plt.xlabel('Opponent')
            plt.ylabel('Main AI')
            
        elif plot_type == 'avg_moves':
            # Average moves comparison
            pivot_df = results_df.pivot(index='main_ai', columns='opponent', values='avg_moves')
            sns.heatmap(pivot_df, annot=True, cmap='RdYlGn_r', fmt='.1f',
                       cbar_kws={'label': 'Average Moves to Win'})
            plt.title('Average Moves to Win Matrix')
            plt.xlabel('Opponent')
            plt.ylabel('Main AI')
            
        elif plot_type == 'performance_summary':
            # Performance summary bar chart
            agent_summary = results_df.groupby('main_ai').agg({
                'win_rate': 'mean',
                'avg_moves': 'mean'
            }).reset_index()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Win rate bar chart
            ax1.bar(agent_summary['main_ai'], agent_summary['win_rate'])
            ax1.set_title('Average Win Rate by AI Agent')
            ax1.set_ylabel('Win Rate')
            ax1.set_xticklabels(agent_summary['main_ai'], rotation=45)
            
            # Average moves bar chart
            ax2.bar(agent_summary['main_ai'], agent_summary['avg_moves'])
            ax2.set_title('Average Moves to Win by AI Agent')
            ax2.set_ylabel('Average Moves')
            ax2.set_xticklabels(agent_summary['main_ai'], rotation=45)
            
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"performance_{plot_type}_{simulation_id}.png"
        plot_path = f"static/plots/{plot_filename}"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return jsonify({
            'success': True,
            'plot_url': f"/static/plots/{plot_filename}"
        })
        
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ================================
# Game State and Monitoring
# ================================

@app.route('/api/game/<game_id>')
def get_game(game_id):
    """Get game state."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game = games[game_id]
    player_name = request.args.get('player_name')
    
    if player_name:
        player_view = game.get_player_view(player_name)
        if player_view is None:
            return jsonify({'error': 'Player not found'}), 404
        return jsonify(player_view)
    else:
        return jsonify(game.to_dict())

@app.route('/api/games')
def list_games():
    """List all active games."""
    game_list = []
    for game_id, game in games.items():
        game_list.append({
            'game_id': game_id,
            'player1': game.player1.name,
            'player2': game.player2.name,
            'game_over': game.is_over(),
            'winner': game.winner.name if game.winner else None,
            'created_at': game.created_at.isoformat()
        })
    return jsonify(game_list)

@app.route('/api/ai_agents')
def list_ai_agents():
    """List all available AI agents."""
    agents = []
    for key, agent_class in AI_AGENTS.items():
        agents.append({
            'key': key,
            'name': agent_class.__name__,
            'description': getattr(agent_class, '__doc__', 'No description available')
        })
    return jsonify(agents)

# ================================
# WebSocket Events
# ================================

@socketio.on('connect')
def on_connect():
    logger.info(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def on_disconnect():
    logger.info(f'Client disconnected: {request.sid}')

@socketio.on('join_game')
def on_join_game(data):
    """Join a game room for real-time updates."""
    game_id = data.get('game_id')
    player_name = data.get('player_name')
    
    if game_id in games:
        join_room(game_id)
        emit('joined_game', {'game_id': game_id, 'player_name': player_name})
        logger.info(f'Player {player_name} joined game {game_id}')

@socketio.on('leave_game')
def on_leave_game(data):
    """Leave a game room."""
    game_id = data.get('game_id')
    leave_room(game_id)
    emit('left_game', {'game_id': game_id})

@socketio.on('join_simulation')
def on_join_simulation(data):
    """Join simulation monitoring room."""
    simulation_id = data.get('simulation_id')
    join_room(f"sim_{simulation_id}")
    emit('joined_simulation', {'simulation_id': simulation_id})

if __name__ == '__main__':
    logger.info("Starting Battleship AI Research Platform Web Server")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)