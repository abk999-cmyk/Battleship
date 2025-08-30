/**
 * Battleship Web Game - Client-side JavaScript
 */

class BattleshipGame {
    constructor() {
        this.socket = io();
        this.gameId = null;
        this.playerName = null;
        this.currentGame = null;
        this.shipPlacements = [];
        this.currentShipSize = null;
        this.currentOrientation = 'H';
        this.shipSizes = [5, 4, 3, 3, 2];
        this.placedShips = [];
        
        this.initializeSocketEvents();
        this.initializeEventListeners();
    }
    
    initializeSocketEvents() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.showAlert('Connection lost. Please refresh the page.', 'warning');
        });
        
        this.socket.on('game_update', (data) => {
            console.log('Game update received:', data);
            this.handleGameUpdate(data);
        });
        
        this.socket.on('ai_move_update', (data) => {
            console.log('AI move update received:', data);
            this.handleAIMoveUpdate(data);
        });
        
        this.socket.on('joined_game', (data) => {
            console.log('Joined game:', data);
        });
    }
    
    initializeEventListeners() {
        // Ship placement orientation change
        document.querySelectorAll('input[name="orientation"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentOrientation = e.target.value;
            });
        });
    }
    
    async createGame() {
        const gameMode = document.getElementById('gameMode').value;
        let gameData = {
            player1_name: document.getElementById('player1Name').value || 'Player 1',
            player2_name: document.getElementById('player2Name').value || 'AI Player',
            game_mode: gameMode
        };

        // Add AI type configurations based on game mode
        if (gameMode === 'human_vs_ai') {
            gameData.ai_type = document.getElementById('aiType1').value;
        } else if (gameMode === 'ai_vs_ai') {
            gameData.ai1_type = document.getElementById('aiType1').value;
            gameData.ai2_type = document.getElementById('aiType2').value;
            gameData.player1_name = document.getElementById('aiType1').selectedOptions[0].text;
            gameData.player2_name = document.getElementById('aiType2').selectedOptions[0].text;
        }
        
        try {
            const response = await fetch('/api/create_game', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(gameData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.gameId = result.game_id;
                this.playerName = gameData.player1_name;
                this.currentGame = result.game;
                
                // Join socket room
                this.socket.emit('join_game', {
                    game_id: this.gameId,
                    player_name: this.playerName
                });
                
                // Hide setup modal
                const setupModal = bootstrap.Modal.getInstance(document.getElementById('gameSetupModal'));
                setupModal.hide();
                
                if (result.needs_ship_placement) {
                    this.showShipPlacementModal();
                } else if (result.is_ai_vs_ai) {
                    this.startAIvsAIGame();
                } else {
                    this.startGame();
                }
                
                this.showAlert('Game created successfully!', 'success');
            } else {
                this.showAlert('Failed to create game: ' + result.error, 'danger');
            }
        } catch (error) {
            console.error('Error creating game:', error);
            this.showAlert('Failed to create game. Please try again.', 'danger');
        }
    }
    
    showShipPlacementModal() {
        const modal = new bootstrap.Modal(document.getElementById('shipPlacementModal'));
        modal.show();
        this.initializeShipPlacement();
    }
    
    initializeShipPlacement() {
        this.placedShips = [];
        this.shipPlacements = [];
        this.currentShipSize = null;
        
        // Create ship placement board
        this.createBoard('shipPlacementBoard', 'placement-board');
        
        // Setup ship selection
        document.querySelectorAll('.ship-item').forEach(item => {
            item.addEventListener('click', () => {
                if (item.classList.contains('placed')) return;
                
                // Remove previous selection
                document.querySelectorAll('.ship-item').forEach(i => i.classList.remove('selected'));
                
                // Select current ship
                item.classList.add('selected');
                this.currentShipSize = parseInt(item.dataset.size);
            });
        });
        
        // Add click listeners to board cells
        document.querySelectorAll('#shipPlacementBoard .board-cell').forEach(cell => {
            cell.addEventListener('click', (e) => this.placeShip(e));
            cell.addEventListener('mouseenter', (e) => this.previewShipPlacement(e));
            cell.addEventListener('mouseleave', () => this.clearPreview());
        });
    }
    
    placeShip(event) {
        if (!this.currentShipSize) {
            this.showAlert('Please select a ship to place first.', 'warning');
            return;
        }
        
        const cell = event.target;
        const row = parseInt(cell.dataset.row);
        const col = parseInt(cell.dataset.col);
        
        const coordinates = this.getShipCoordinates(row, col, this.currentShipSize, this.currentOrientation);
        
        if (!this.isValidPlacement(coordinates)) {
            this.showAlert('Invalid ship placement. Try a different position.', 'warning');
            return;
        }
        
        // Place the ship
        this.placedShips.push({
            size: this.currentShipSize,
            coordinates: coordinates,
            orientation: this.currentOrientation
        });
        
        // Update board display
        coordinates.forEach(([r, c]) => {
            const cellElement = document.querySelector(`#shipPlacementBoard .board-cell[data-row="${r}"][data-col="${c}"]`);
            cellElement.classList.add('placed-ship');
        });
        
        // Mark ship as placed
        const shipItem = document.querySelector(`.ship-item.selected`);
        shipItem.classList.add('placed');
        shipItem.classList.remove('selected');
        
        // Find next unplaced ship
        const nextShip = document.querySelector('.ship-item:not(.placed)');
        if (nextShip) {
            nextShip.classList.add('selected');
            this.currentShipSize = parseInt(nextShip.dataset.size);
        } else {
            this.currentShipSize = null;
            document.getElementById('confirmPlacementBtn').disabled = false;
        }
        
        this.clearPreview();
    }
    
    previewShipPlacement(event) {
        if (!this.currentShipSize) return;
        
        this.clearPreview();
        
        const cell = event.target;
        const row = parseInt(cell.dataset.row);
        const col = parseInt(cell.dataset.col);
        
        const coordinates = this.getShipCoordinates(row, col, this.currentShipSize, this.currentOrientation);
        const isValid = this.isValidPlacement(coordinates);
        
        coordinates.forEach(([r, c]) => {
            const cellElement = document.querySelector(`#shipPlacementBoard .board-cell[data-row="${r}"][data-col="${c}"]`);
            if (cellElement) {
                cellElement.classList.add(isValid ? 'preview' : 'invalid-preview');
            }
        });
    }
    
    clearPreview() {
        document.querySelectorAll('#shipPlacementBoard .board-cell').forEach(cell => {
            cell.classList.remove('preview', 'invalid-preview');
        });
    }
    
    getShipCoordinates(row, col, size, orientation) {
        const coordinates = [];
        
        for (let i = 0; i < size; i++) {
            if (orientation === 'H') {
                coordinates.push([row, col + i]);
            } else {
                coordinates.push([row + i, col]);
            }
        }
        
        return coordinates;
    }
    
    isValidPlacement(coordinates) {
        // Check bounds
        for (const [row, col] of coordinates) {
            if (row < 0 || row >= 10 || col < 0 || col >= 10) {
                return false;
            }
        }
        
        // Check for overlaps with existing ships
        for (const ship of this.placedShips) {
            for (const [row, col] of ship.coordinates) {
                for (const [newRow, newCol] of coordinates) {
                    if (row === newRow && col === newCol) {
                        return false;
                    }
                }
            }
        }
        
        return true;
    }
    
    randomPlaceShips() {
        this.clearShips();
        
        for (const size of this.shipSizes) {
            let placed = false;
            let attempts = 0;
            
            while (!placed && attempts < 100) {
                attempts++;
                const row = Math.floor(Math.random() * 10);
                const col = Math.floor(Math.random() * 10);
                const orientation = Math.random() < 0.5 ? 'H' : 'V';
                
                const coordinates = this.getShipCoordinates(row, col, size, orientation);
                
                if (this.isValidPlacement(coordinates)) {
                    this.placedShips.push({
                        size: size,
                        coordinates: coordinates,
                        orientation: orientation
                    });
                    
                    coordinates.forEach(([r, c]) => {
                        const cellElement = document.querySelector(`#shipPlacementBoard .board-cell[data-row="${r}"][data-col="${c}"]`);
                        cellElement.classList.add('placed-ship');
                    });
                    
                    placed = true;
                }
            }
        }
        
        // Mark all ships as placed
        document.querySelectorAll('.ship-item').forEach(item => {
            item.classList.add('placed');
            item.classList.remove('selected');
        });
        
        this.currentShipSize = null;
        document.getElementById('confirmPlacementBtn').disabled = false;
    }
    
    clearShips() {
        this.placedShips = [];
        document.querySelectorAll('#shipPlacementBoard .board-cell').forEach(cell => {
            cell.classList.remove('placed-ship');
        });
        
        document.querySelectorAll('.ship-item').forEach(item => {
            item.classList.remove('placed', 'selected');
        });
        
        document.getElementById('confirmPlacementBtn').disabled = true;
        this.currentShipSize = null;
    }
    
    async confirmShipPlacement() {
        try {
            const response = await fetch('/api/place_ships', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    game_id: this.gameId,
                    player_name: this.playerName,
                    ship_placements: this.placedShips
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                const modal = bootstrap.Modal.getInstance(document.getElementById('shipPlacementModal'));
                modal.hide();
                this.startGame();
                this.showAlert('Ships placed successfully!', 'success');
            } else {
                this.showAlert('Failed to place ships: ' + result.error, 'danger');
            }
        } catch (error) {
            console.error('Error placing ships:', error);
            this.showAlert('Failed to place ships. Please try again.', 'danger');
        }
    }
    
    startGame() {
        document.getElementById('welcomeScreen').style.display = 'none';
        document.getElementById('gameArea').style.display = 'block';
        
        this.createBoard('attackBoard', 'attack-board');
        this.createBoard('defenseBoard', 'defense-board');
        
        this.loadGameState();
    }
    
    startAIvsAIGame() {
        document.getElementById('welcomeScreen').style.display = 'none';
        document.getElementById('gameArea').style.display = 'block';
        
        // Create boards first
        this.createBoard('attackBoard', 'spectator-board');
        this.createBoard('defenseBoard', 'spectator-board');
        
        // Update UI for spectator mode after boards are created
        setTimeout(() => {
            try {
                const attackHeader = document.querySelector('#attackBoard').parentElement.querySelector('.card-header h5');
                const attackSubtext = document.querySelector('#attackBoard').parentElement.querySelector('.card-header small');
                const defenseHeader = document.querySelector('#defenseBoard').parentElement.querySelector('.card-header h5');
                const defenseSubtext = document.querySelector('#defenseBoard').parentElement.querySelector('.card-header small');
                
                if (attackHeader) attackHeader.innerHTML = '<i class="fas fa-eye"></i> AI Battle View';
                if (attackSubtext) attackSubtext.textContent = 'AI vs AI - Spectator Mode';
                if (defenseHeader) defenseHeader.innerHTML = '<i class="fas fa-eye"></i> AI Ships & Attacks';
                if (defenseSubtext) defenseSubtext.textContent = 'Watch the AI battle';
            } catch (error) {
                console.log('UI update error (non-critical):', error);
            }
        }, 100);
        
        this.loadGameState();
        
        // Show spectator message
        this.showAlert('AI vs AI battle starting! Watch the intelligent agents fight...', 'info');
    }
    
    createBoard(containerId, className) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Board container ${containerId} not found`);
            return;
        }
        
        container.innerHTML = '';
        container.className = `battleship-board ${className}`;
        
        // Create the 10x10 grid
        for (let row = 0; row < 10; row++) {
            for (let col = 0; col < 10; col++) {
                const cell = document.createElement('div');
                cell.className = 'board-cell water';
                cell.dataset.row = row;
                cell.dataset.col = col;
                
                // Add click handlers based on board type
                if (className === 'attack-board') {
                    cell.addEventListener('click', (e) => this.makeAttack(e));
                    cell.style.cursor = 'crosshair';
                } else if (className === 'spectator-board') {
                    cell.style.cursor = 'default';
                } else if (className === 'defense-board') {
                    cell.style.cursor = 'default';
                }
                
                container.appendChild(cell);
            }
        }
        
        console.log(`Created ${className} board with ${container.children.length} cells`);
    }
    
    async makeAttack(event) {
        const cell = event.target;
        const row = parseInt(cell.dataset.row);
        const col = parseInt(cell.dataset.col);
        
        if (cell.classList.contains('hit') || cell.classList.contains('miss')) {
            this.showAlert('You already attacked this position!', 'warning');
            return;
        }
        
        try {
            const response = await fetch('/api/make_move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    game_id: this.gameId,
                    player_name: this.playerName,
                    coord: [row, col]
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.updateAttackBoard(row, col, result.result);
                this.updateGameInfo(result);
                
                if (result.game_over) {
                    this.handleGameEnd(result.winner);
                }
            } else {
                this.showAlert('Failed to make move: ' + result.error, 'danger');
            }
        } catch (error) {
            console.error('Error making attack:', error);
            this.showAlert('Failed to make attack. Please try again.', 'danger');
        }
    }
    
    updateAttackBoard(row, col, result) {
        const cell = document.querySelector(`#attackBoard .board-cell[data-row="${row}"][data-col="${col}"]`);
        cell.classList.remove('water');
        cell.classList.add(result);
    }
    
    updateDefenseBoard(gameState) {
        const playerView = gameState.player || gameState;
        const board = playerView.board;
        
        if (board && board.grid) {
            board.grid.forEach((row, rowIndex) => {
                row.forEach((cellState, colIndex) => {
                    const cell = document.querySelector(`#defenseBoard .board-cell[data-row="${rowIndex}"][data-col="${colIndex}"]`);
                    if (cell) {
                        cell.className = `board-cell ${cellState}`;
                    }
                });
            });
        }
    }
    
    updateGameInfo(gameData) {
        document.getElementById('currentTurn').textContent = gameData.current_player || 'Unknown';
        document.getElementById('moveCount').textContent = gameData.move_count || 0;
        document.getElementById('lastMove').textContent = gameData.last_move ? 
            `(${gameData.last_move[0]}, ${gameData.last_move[1]})` : '-';
        
        const statusElement = document.getElementById('gameStatus');
        if (gameData.game_over) {
            statusElement.textContent = 'Game Over';
            statusElement.className = 'badge bg-danger fs-6';
        } else {
            statusElement.textContent = 'Active';
            statusElement.className = 'badge bg-success fs-6';
        }
    }
    
    async loadGameState() {
        try {
            const response = await fetch(`/api/game/${this.gameId}?player_name=${this.playerName}`);
            const gameState = await response.json();
            
            if (gameState.error) {
                this.showAlert('Failed to load game state: ' + gameState.error, 'danger');
                return;
            }
            
            this.updateDefenseBoard(gameState);
            this.updateGameInfo(gameState);
            
            // Update attack board based on target grid
            if (gameState.opponent && gameState.opponent.target_grid) {
                gameState.opponent.target_grid.forEach((row, rowIndex) => {
                    row.forEach((cellState, colIndex) => {
                        const cell = document.querySelector(`#attackBoard .board-cell[data-row="${rowIndex}"][data-col="${colIndex}"]`);
                        if (cell) {
                            if (cellState === 1) {
                                cell.classList.add('miss');
                            } else if (cellState === 2) {
                                cell.classList.add('hit');
                            } else if (cellState === 3) {
                                cell.classList.add('sunk');
                            }
                        }
                    });
                });
            }
            
        } catch (error) {
            console.error('Error loading game state:', error);
            this.showAlert('Failed to load game state.', 'danger');
        }
    }
    
    handleGameUpdate(data) {
        if (data.game_id !== this.gameId) return;
        
        if (data.move_result) {
            this.updateGameInfo(data.move_result);
            
            if (data.move_result.game_over) {
                this.handleGameEnd(data.move_result.winner);
            }
        }
        
        if (data.game_state) {
            this.loadGameState();
        }
    }
    
    handleAIMoveUpdate(data) {
        console.log('AI move update:', data);
        if (data.game_id !== this.gameId) return;
        
        const [row, col] = data.move;
        
        // Show the AI move on the attack board
        const attackCell = document.querySelector(`#attackBoard .board-cell[data-row="${row}"][data-col="${col}"]`);
        if (attackCell) {
            attackCell.classList.remove('water');
            attackCell.classList.add(data.result);
            
            // Add animation effect
            attackCell.classList.add('bounce');
            setTimeout(() => attackCell.classList.remove('bounce'), 500);
        }
        
        // Update both boards from game state
        this.updateBoardsFromGameState(data.game_state);
        
        // Update game info
        this.updateGameInfo({
            current_player: data.game_state.current_player || data.player,
            move_count: data.game_state.move_count || 0,
            last_move: data.move,
            last_result: data.result
        });
        
        // Show move notification
        const moveText = `${data.player} attacks (${row}, ${col}) → ${data.result.toUpperCase()}!`;
        this.showMoveNotification(moveText, data.result);
        
        // Check for game end
        if (data.game_over) {
            setTimeout(() => {
                this.handleGameEnd(data.winner);
            }, 1000);
        }
    }
    
    updateBoardsFromGameState(gameState) {
        // This method updates both boards based on the game state
        if (!gameState) return;
        
        try {
            // Update attack board (showing moves made)
            for (let row = 0; row < 10; row++) {
                for (let col = 0; col < 10; col++) {
                    const attackCell = document.querySelector(`#attackBoard .board-cell[data-row="${row}"][data-col="${col}"]`);
                    const defenseCell = document.querySelector(`#defenseBoard .board-cell[data-row="${row}"][data-col="${col}"]`);
                    
                    // For now, just ensure cells exist
                    if (attackCell && !attackCell.classList.contains('hit') && !attackCell.classList.contains('miss')) {
                        // Keep as water unless marked otherwise
                    }
                    
                    if (defenseCell && !defenseCell.classList.contains('hit') && !defenseCell.classList.contains('miss')) {
                        // Keep as water unless marked otherwise
                    }
                }
            }
        } catch (error) {
            console.error('Error updating boards from game state:', error);
        }
    }
    
    showMoveNotification(text, result) {
        const notification = document.createElement('div');
        notification.className = `alert alert-${result === 'hit' ? 'success' : result === 'miss' ? 'info' : 'warning'} position-fixed`;
        notification.style.cssText = 'top: 80px; right: 20px; z-index: 1060; max-width: 300px;';
        notification.innerHTML = `<strong>AI Move:</strong> ${text}`;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 2500);
    }
    
    handleGameEnd(winner) {
        const gameArea = document.getElementById('gameArea');
        const winnerBanner = document.createElement('div');
        winnerBanner.className = 'winner-banner fade-in';
        
        if (winner === this.playerName) {
            winnerBanner.innerHTML = '<h2>🎉 Congratulations! You Won! 🎉</h2>';
        } else {
            winnerBanner.innerHTML = `<h2>💔 Game Over! ${winner} Won! 💔</h2>`;
        }
        
        gameArea.insertBefore(winnerBanner, gameArea.firstChild);
        
        // Disable attack board
        document.querySelectorAll('#attackBoard .board-cell').forEach(cell => {
            cell.classList.add('disabled');
            cell.style.pointerEvents = 'none';
        });
    }
    
    newGame() {
        // Reset game state
        this.gameId = null;
        this.playerName = null;
        this.currentGame = null;
        
        // Show welcome screen
        document.getElementById('gameArea').style.display = 'none';
        document.getElementById('welcomeScreen').style.display = 'block';
        
        // Show setup modal
        const setupModal = new bootstrap.Modal(document.getElementById('gameSetupModal'));
        setupModal.show();
    }
    
    endGame() {
        if (confirm('Are you sure you want to end the current game?')) {
            this.newGame();
        }
    }
    
    showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alertContainer');
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        alertContainer.appendChild(alert);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    }
}

// Global functions for HTML onclick handlers
let game;

function createGame() {
    game.createGame();
}

function randomPlaceShips() {
    game.randomPlaceShips();
}

function clearShips() {
    game.clearShips();
}

function confirmShipPlacement() {
    game.confirmShipPlacement();
}

function newGame() {
    game.newGame();
}

function endGame() {
    game.endGame();
}

// Initialize game when page loads
document.addEventListener('DOMContentLoaded', function() {
    game = new BattleshipGame();
    
    // Show setup modal initially
    const setupModal = new bootstrap.Modal(document.getElementById('gameSetupModal'));
    setupModal.show();
});
