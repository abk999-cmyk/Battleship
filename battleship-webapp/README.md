# Battleship Web Application

A modern web-based version of the classic Battleship game with AI opponents, built with Flask and real-time WebSocket communication.

## 🎯 Features

- **Interactive Web Interface**: Modern, responsive design with Bootstrap
- **Real-time Gameplay**: WebSocket-powered live updates
- **AI Opponents**: Multiple difficulty levels (Easy, Medium, Hard)
- **Game Modes**: 
  - Human vs AI
  - AI vs AI (Watch mode)
  - Human vs Human
- **Ship Placement**: Manual placement or random auto-placement
- **Game Dashboard**: Statistics and game management
- **Mobile Responsive**: Works on desktop, tablet, and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Navigate to the webapp directory:**
```bash
cd battleship-webapp
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python3 app.py
```

4. **Open your browser and visit:**
```
http://localhost:5000
```

## 🎮 How to Play

1. **Start a New Game**: Click "Start New Game" on the welcome screen
2. **Choose Game Mode**: Select from Human vs AI, AI vs AI, or Human vs Human
3. **Set AI Difficulty**: Choose Easy, Medium, or Hard for AI opponents
4. **Place Your Ships**: 
   - Manual placement: Click on the board to place ships
   - Auto placement: Use "Random Placement" button
5. **Play**: Click on the opponent's board to make attacks
6. **Win**: Sink all enemy ships to win!

## 🏗️ Project Structure

```
battleship-webapp/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── game_logic/           # Core game engine
│   ├── __init__.py
│   ├── ship.py          # Ship class
│   ├── board.py         # Game board logic
│   ├── player.py        # Player classes (Human/AI)
│   └── game.py          # Game engine
├── templates/            # HTML templates
│   ├── base.html        # Base template
│   ├── index.html       # Main game interface
│   └── dashboard.html   # Game statistics
└── static/              # Static assets
    ├── css/
    │   └── style.css    # Game styles
    ├── js/
    │   └── game.js      # Client-side game logic
    └── img/             # Game images (auto-generated)
```

## 🤖 AI Difficulty Levels

- **Easy**: Random attacks with basic targeting
- **Medium**: Smart hunting mode with probability-based targeting
- **Hard**: Advanced strategies with ship prediction algorithms

## 🌐 API Endpoints

- `POST /api/create_game` - Create a new game
- `POST /api/place_ships` - Place ships on the board
- `POST /api/make_move` - Make an attack move
- `GET /api/game/<id>` - Get game state
- `GET /api/games` - List all games

## 🔌 WebSocket Events

- `connect/disconnect` - Client connection management
- `join_game/leave_game` - Game room management
- `game_update` - Real-time game state updates

## 🎨 Technologies Used

- **Backend**: Flask, Flask-SocketIO, Python
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Real-time**: WebSocket with Socket.IO
- **Game Logic**: Custom Python classes with NumPy

## 🐛 Troubleshooting

### Port Already in Use
If port 5000 is busy, you can change it in `app.py`:
```python
socketio.run(app, debug=True, host='0.0.0.0', port=8080)
```

### Import Errors
Make sure you're in the webapp directory and all dependencies are installed:
```bash
cd battleship-webapp
pip install -r requirements.txt
```

### Browser Connection Issues
- Check firewall settings
- Try accessing via `127.0.0.1:5000` instead of `localhost:5000`
- Ensure no ad blockers are interfering with WebSocket connections

## 🔧 Development

### Running in Development Mode
The app runs in debug mode by default, which includes:
- Auto-reload on file changes
- Detailed error messages
- Debug console access

### Adding New Features
1. Game logic changes: Modify files in `game_logic/`
2. UI changes: Update templates and static files
3. API changes: Modify `app.py`

## 📱 Mobile Support

The web app is fully responsive and works on:
- Desktop browsers (Chrome, Firefox, Safari, Edge)
- Tablet devices (iPad, Android tablets)
- Mobile phones (iOS Safari, Android Chrome)

## 🎯 Game Rules

Standard Battleship rules apply:
- **Fleet**: Carrier (5), Battleship (4), Cruiser (3), Submarine (3), Destroyer (2)
- **Objective**: Sink all enemy ships
- **Turns**: Players alternate making attacks
- **Hits**: Continue attacking until you miss
- **Victory**: First player to sink all enemy ships wins

---

Enjoy playing Battleship on the web! 🚢⚓
