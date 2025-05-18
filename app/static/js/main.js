// --- START OF FILE main.js ---

document.addEventListener('DOMContentLoaded', () => {
    // --- Game Elements ---
    const boardElement = document.querySelector('.board');
    let cells = document.querySelectorAll('.cell');
    const resetButton = document.getElementById('reset');
    const undoButton = document.getElementById('undo');
    const soundToggle = document.getElementById('sound-toggle');
    const playerCountdownBar = document.getElementById('player-countdown');
    const aiCountdownBar = document.getElementById('ai-countdown');
    const playerScoreDisplay = document.getElementById('player-score');
    const aiScoreDisplay = document.getElementById('ai-score');
    const difficultySelect = document.getElementById('difficulty');
    const algorithmSelect = document.getElementById('algorithm');
    const visualizationToggleButton = document.getElementById('toggle-visualization'); // Added

    // --- Sound Effects ---
    const sounds = {};
    if (typeof soundPaths !== 'undefined') {
        Object.keys(soundPaths).forEach(key => {
            if (soundPaths[key]) {
                try {
                    sounds[key] = new Audio(soundPaths[key]);
                    sounds[key].load();
                    sounds[key].volume = 0.5;
                } catch (e) {
                    console.error(`Error loading sound ${key}: ${e}`);
                }
            }
        });
    } else {
        console.warn("Sound paths (soundPaths object) not defined in HTML.");
    }

    // --- Game State ---
    let boardState = Array(6).fill(null).map(() => Array(7).fill(null));
    let currentPlayer = 'player';
    let gameActive = true;
    let isAnimating = false;
    let moveHistory = [];
    let currentCells = cells;

    // --- Visualization State (Added) ---
    let visualizationActive = false;
    let moveProbabilities = Array(7).fill(0);

    
    // --- Difficulty & Stats State (Added) ---
    // Note: Base settings are still used for *suggestions*, but actual AI adaptation happens backend-side
    const difficultySettings = {
        easy: { depth: 2, randomness: 0.7 },
        medium: { depth: 4, randomness: 0.4 },
        hard: { depth: 6, randomness: 0.1 },
        expert: { depth: 8, randomness: 0 }
    };
    let playerStats = {
        gamesPlayed: 0,
        wins: 0,
        avgMovesToWin: 0,
        lastMoves: []
    };

    // --- Sound Functions ---
    function playSound(type) {
        const currentSoundToggle = document.getElementById('sound-toggle');
        if (!currentSoundToggle || !currentSoundToggle.checked) return;

        const sound = sounds[type];
        if (sound) {
            sound.currentTime = 0;
            sound.play().catch(e => console.warn(`Sound play failed for ${type}: ${e.message}`));
        } else {
            console.warn(`Sound type "${type}" not found or loaded.`);
        }
    }

    // --- Initialization ---
     // Fix for metrics buttons - don't clone them as they're not part of the game board
    const refreshBtn = document.getElementById('refresh-metrics');
    const resetBtn = document.getElementById('reset-metrics');
    const exportBtn = document.getElementById('export-metrics');
    
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadMetrics);
    }
    if (resetBtn) {
        resetBtn.addEventListener('click', resetMetrics);
    }
    if (exportBtn) {
        exportBtn.addEventListener('click', exportMetrics);
    }

    // --- Event Handlers ---
    function handlePlayerMove(col) {
        if (!gameActive || currentPlayer !== 'player' || isAnimating) {
            console.log(`Move blocked: gameActive=${gameActive}, currentPlayer=${currentPlayer}, isAnimating=${isAnimating}`);
            playSound('error');
            return;
        }

        const row = findEmptyRow(col);
        if (row === -1) {
            console.log(`Column ${col} is full.`);
            playSound('error');
            const columnCells = boardElement.querySelectorAll(`.cell[data-col="${col}"]`);
            columnCells.forEach(c => c.classList.add('shake'));
            setTimeout(() => columnCells.forEach(c => c.classList.remove('shake')), 300);
            return;
        }

        makeMove(row, col, 'player');
    }

    // --- Core Game Logic ---
    function makeMove(row, col, player) {
        if (boardState[row][col] !== null) {
            console.warn(`Attempted move on occupied cell [${row}, ${col}]`);
            return;
        }

        console.log(`makeMove: ${player} at [${row}, ${col}]`);
        isAnimating = true;
        gameActive = false; // Temporarily disable interaction during animation/AI turn

        boardState[row][col] = player;
        moveHistory.push({ row, col, player });

        playSound('drop');

        // Clear heatmap before animating player move
        if (player === 'player' && visualizationActive) {
             moveProbabilities = Array(7).fill(0);
             updateHeatmap();
        }

        animateMove(row, col, player, () => {
            console.log(`Animation finished for ${player} at [${row}, ${col}]`);
            isAnimating = false;

            if (checkWin(row, col, player)) {
                console.log(`${player} wins!`);
                playSound('win');
                endGame(player); // Pass the actual winner
                return;
            }

            if (isBoardFull()) {
                console.log("It's a draw!");
                playSound('error');
                endGame(null); // Pass null for a draw
                return;
            }

            currentPlayer = player === 'player' ? 'ai' : 'player';
            console.log(`Switching turn to: ${currentPlayer}`);
            gameActive = true; // Re-enable game logic

            updateUI(); // Update UI to reflect new player turn

            if (currentPlayer === 'ai' && gameActive) {
                console.log("Triggering AI move...");
                // Optional: Predict moves immediately if visualization is on
                if (visualizationActive) {
                     predictAIMoves(); // Show prediction before the thinking delay
                }
                setTimeout(makeAIMove, 600); // Delay for AI 'thinking' time
            }
        });
    }


    function makeAIMove() {
        if (!gameActive || currentPlayer !== 'ai' || isAnimating) {
            console.log("AI move aborted:", { gameActive, currentPlayer, isAnimating });
            if (!isAnimating && !gameActive) {
                currentPlayer = 'player';
                gameActive = true;
                updateUI();
            }
            return;
        }
    
        console.log("AI is thinking...");
        isAnimating = true;
        gameActive = false;
        updateUI();
    
        // Remove existing indicators
        document.querySelectorAll('.ai-thinking-indicator').forEach(ind => ind.remove());
        
        const thinkingIndicator = document.createElement('div');
        thinkingIndicator.className = 'ai-thinking-indicator';
        thinkingIndicator.textContent = `AI (${algorithmSelect.value}) is thinking...`;
        document.querySelector('.game-container').appendChild(thinkingIndicator);
    
        // Calculate win rate for adaptation
        const winRate = playerStats.gamesPlayed > 0 ? playerStats.wins / playerStats.gamesPlayed : 0.5;
    
        // Make the API request with proper error handling
        fetch('/api/get_ai_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                board: boardState,
                algorithm: algorithmSelect.value,
                difficulty: difficultySelect.value,
                win_rate: winRate,
                get_probabilities: visualizationActive
            })
        })
        .then(response => {
            if (!response.ok) {
                // Try to get error message from response
                return response.text().then(text => {
                    throw new Error(text || `HTTP error! status: ${response.status}`);
                });
            }
            return response.text().then(text => {
                try {
                    return JSON.parse(text);
                } catch (e) {
                    console.error('Failed to parse JSON:', text);
                    throw new Error('Invalid JSON response from server');
                }
            });
        })
        .then(data => {
            if (!data) {
                throw new Error('No data received from server');
            }
          loadMetrics(); // Refresh metrics after move
            console.log("AI Response:", {
                move: data.move,
                algorithm: data.algorithm,
                status: data.status,
                rawResponse: data.raw_response || 'N/A'
            });
    
            document.querySelector('.ai-thinking-indicator')?.remove();
    
            // Update heatmap if visualization is active
            if (visualizationActive && data.probabilities) {
                moveProbabilities = data.probabilities;
                updateHeatmap();
            }
    
            if (data.move === undefined || data.move === -1) {
                throw new Error('AI did not return a valid move');
            }
    
            const col = data.move;
            const row = findEmptyRow(col);
            if (row === -1) {
                throw new Error(`AI chose invalid column ${col} (column is full)`);
            }
    
            makeMove(row, col, 'ai');
        })
        .catch(error => {
            console.error('AI Move Error:', error);
            document.querySelector('.ai-thinking-indicator')?.remove();
            
            // Fallback to random move if Deepseek fails
            if (algorithmSelect.value === 'deepseek') {
                console.warn('Falling back to random AI due to Deepseek error');
                const validMoves = [...Array(7).keys()].filter(col => boardState[0][col] === null);
                if (validMoves.length > 0) {
                    const col = validMoves[Math.floor(Math.random() * validMoves.length)];
                    const row = findEmptyRow(col);
                    makeMove(row, col, 'ai');
                    return;
                }
            }
    
            currentPlayer = 'player';
            isAnimating = false;
            gameActive = true;
            updateUI();
            
            alert(`Error communicating with AI: ${error.message}. It's your turn again.`);
        });
    }

    // --- Helper Functions ---

    function findEmptyRow(col) {
        if (col < 0 || col >= 7) return -1;
        for (let row = 5; row >= 0; row--) {
            if (boardState[row][col] === null) return row;
        }
        return -1; // Column is full
    }

    function isBoardFull() {
        // Check only the top row
        for (let col = 0; col < 7; col++) {
            if (boardState[0][col] === null) {
                return false; // Found an empty slot in the top row
            }
        }
        return true; // Top row is full, so the board is full
    }

    function checkWin(row, col, player) {
        const numRows = 6;
        const numCols = 7;

        // Directions: Horizontal, Vertical, Diagonal Down-Right, Diagonal Up-Right
        const directions = [
            { dr: 0, dc: 1 }, // Horizontal
            { dr: 1, dc: 0 }, // Vertical
            { dr: 1, dc: 1 }, // Diagonal Down-Right
            { dr: 1, dc: -1 } // Diagonal Up-Right (checks Down-Left implicitly)
        ];

        for (const { dr, dc } of directions) {
            let count = 1; // Count the token just placed

            // Check in the positive direction (e.g., right, down, down-right, up-right)
            for (let i = 1; i < 4; i++) {
                const r = row + i * dr;
                const c = col + i * dc;
                if (r >= 0 && r < numRows && c >= 0 && c < numCols && boardState[r][c] === player) {
                    count++;
                } else {
                    break; // Stop checking in this direction
                }
            }

            // Check in the negative direction (e.g., left, up, up-left, down-left)
            for (let i = 1; i < 4; i++) {
                const r = row - i * dr;
                const c = col - i * dc;
                if (r >= 0 && r < numRows && c >= 0 && c < numCols && boardState[r][c] === player) {
                    count++;
                } else {
                    break; // Stop checking in this direction
                }
            }

            if (count >= 4) {
                console.log(`Win detected for ${player} at [${row}, ${col}] with direction {dr:${dr}, dc:${dc}}`);
                return true; // Found a win
            }
        }
        return false; // No win found for this move
    }

    // --- Animation ---
    function animateMove(row, col, player, callback) {
        const targetCellSelector = `.cell[data-row="${Number(row)}"][data-col="${Number(col)}"]`;
        const cell = boardElement.querySelector(targetCellSelector);

        if (!cell) {
            console.error(`Cannot find cell for animation: ${targetCellSelector}`);
            if (callback) callback();
            return;
        }

        const token = document.createElement('div');
        token.className = `token ${player}`;

        // --- Calculate Drop Animation ---
        const boardRect = boardElement.getBoundingClientRect();
        const cellRect = cell.getBoundingClientRect();

        // Start position: Centered horizontally above the target column, just above the board
        const startX = cellRect.left - boardRect.left;
        const startY = 0 - cellRect.height * 1.5; // Start slightly above the board view
        const endY = cellRect.top - boardRect.top; // Final position within the cell

        token.style.position = 'absolute';
        token.style.left = `${startX}px`;
        token.style.top = `${startY}px`;
        token.style.width = `${cellRect.width}px`; // Match cell size
        token.style.height = `${cellRect.height}px`;// Match cell size
        token.style.zIndex = '10'; // Ensure token is above the grid

        boardElement.appendChild(token);

        // Force reflow to apply initial styles before transition
        void token.offsetWidth;

        // Apply transition for the drop
        token.style.transition = 'top 0.4s cubic-bezier(0.6, -0.28, 0.74, 0.05)'; // Ease-in with bounce effect
        token.style.top = `${endY}px`;

        // --- Callback on Transition End ---
        token.addEventListener('transitionend', () => {
            // Clean up absolute positioning styles
            token.style.position = '';
            token.style.left = '';
            token.style.top = '';
            token.style.transition = '';
            token.style.zIndex = '';
            token.style.width = '';  // Let CSS handle size now
            token.style.height = ''; // Let CSS handle size now

            // Append the token to the target cell for proper layout
            cell.appendChild(token);

            // Execute the callback function (checks for win/draw, switches player, etc.)
            if (callback) {
                console.log(`Animation callback executing for [${row}, ${col}]`);
                callback();
            } else {
                console.log(`Animation ended, no callback provided for [${row}, ${col}]`);
            }
        }, { once: true }); // Ensure the event listener runs only once
    }

    // --- Game State Management ---
    function endGame(winner) {
        console.log(`Game ended. Winner: ${winner}`);
        gameActive = false; // Stop further moves
        isAnimating = false; // Ensure animation state is cleared
        moveProbabilities = Array(7).fill(0); // Clear heatmap on game end
        updateHeatmap();

        const currentScorePlayer = document.getElementById('player-score');
        const currentScoreAI = document.getElementById('ai-score');
        let playerCurrent = parseInt(currentScorePlayer.textContent.split(': ')[1] || '0');
        let aiCurrent = parseInt(currentScoreAI.textContent.split(': ')[1] || '0');

        // --- Update Player Stats (Added) ---
        playerStats.gamesPlayed++;
        if (winner === 'player') {
            playerStats.wins++;
            // Calculate running average for moves to win (only if wins > 0)
             if (playerStats.wins > 0) {
                const totalWinMoves = (playerStats.avgMovesToWin * (playerStats.wins - 1)) + moveHistory.length;
                 playerStats.avgMovesToWin = totalWinMoves / playerStats.wins;
             } else {
                 playerStats.avgMovesToWin = moveHistory.length; // First win
             }
        }
        playerStats.lastMoves.push(moveHistory.length);
        if (playerStats.lastMoves.length > 10) { // Keep only last 10 game lengths
            playerStats.lastMoves.shift();
        }
        console.log("Player Stats Updated:", playerStats);
        // --- End Stats Update ---

        // Update score display and show result message
        if (winner === 'player') {
            currentScorePlayer.textContent = `Player: ${playerCurrent + 1}`;
            setTimeout(() => alert('You win!'), 150); // Short delay for animation/sound
        } else if (winner === 'ai') {
            currentScoreAI.textContent = `AI: ${aiCurrent + 1}`;
            setTimeout(() => alert('AI wins!'), 150);
        } else { // Draw
            setTimeout(() => alert("It's a draw!"), 150);
        }

        // --- Suggest Difficulty Change (Added) ---
        updateDifficulty(); // Call this to trigger suggestion logic after game ends

        updateUI(); // Update UI to reflect game over state
    }

    function resetGame() {
        console.log("Resetting game...");
        playSound('reset');
        
        // Reset all game state variables
        boardState = Array(6).fill(null).map(() => Array(7).fill(null));
        currentPlayer = 'player';
        gameActive = true;
        isAnimating = false;
        moveHistory = [];
        moveProbabilities = Array(7).fill(0);
        
        // Clear all tokens from the board
        document.querySelectorAll('.token').forEach(token => token.remove());
        
        // Reset UI elements
        updateUI();
        
        // Reset visualization if active
        if (visualizationActive) {
            visualizationActive = false;
            document.querySelector('.board').classList.remove('heatmap-visible');
            document.querySelector('.heatmap-key').style.opacity = 0;
            document.getElementById('toggle-visualization').textContent = 'Show AI Heatmap';
            updateHeatmap();
        }
        
        // Reattach event listeners to cells (in case they were cloned)
        currentCells = document.querySelectorAll('.cell');
        currentCells.forEach(cell => {
            cell.removeEventListener('click', handlePlayerMove);
            cell.addEventListener('click', () => handlePlayerMove(parseInt(cell.dataset.col)));
        });
        
        console.log("Game fully reset. Current state:", {
            boardState,
            currentPlayer,
            gameActive,
            isAnimating
        });
    }

    function undoMove() {
        const currentUndoButton = document.getElementById('undo'); // Get current button instance

        if (moveHistory.length === 0 || isAnimating || !gameActive) { // Also check gameActive here
            console.log("Undo blocked: No history or game/animation is inactive.");
            playSound('error');
            currentUndoButton.disabled = true; // Should be handled by updateUI, but good failsafe
            return;
        }

        // Determine how many moves to undo (usually player + AI = 2, unless player just moved)
        let movesToUndo = 0;
        if (moveHistory.length > 0) {
            // If the last move was AI, undo AI and Player's previous move (2 moves)
            if (moveHistory[moveHistory.length - 1].player === 'ai') {
                 movesToUndo = (moveHistory.length >= 2) ? 2 : 1; // Undo 2 if possible
            } else {
                 // If last move was player, undo just that move (1 move)
                 movesToUndo = 1;
            }
        }


        console.log(`Attempting to undo ${movesToUndo} move(s)...`);
        playSound('drop'); // Use drop sound for undo removal effect

        for (let i = 0; i < movesToUndo; i++) {
            if (moveHistory.length === 0) break; // Stop if history becomes empty

            const lastMove = moveHistory.pop();
            if (lastMove) {
                console.log(`Undoing move: ${lastMove.player} at [${lastMove.row}, ${lastMove.col}]`);
                boardState[lastMove.row][lastMove.col] = null; // Clear the board state

                // Find the cell and remove the corresponding token
                const cellSelector = `.cell[data-row="${lastMove.row}"][data-col="${lastMove.col}"]`;
                const cell = boardElement.querySelector(cellSelector);
                if (cell) {
                    // Find the specific player's token within the cell
                    const token = cell.querySelector(`.token.${lastMove.player}`);
                    if (token) {
                        token.remove();
                        console.log(`Removed token from ${cellSelector}`);
                    } else {
                        // This might happen if the DOM was reset weirdly, log a warning
                        console.warn(`Could not find token for ${lastMove.player} in ${cellSelector} during undo.`);
                    }
                } else {
                    console.warn(`Could not find cell ${cellSelector} during undo.`);
                }
            }
        }

        // Reset game state after undo
        currentPlayer = 'player'; // Always player's turn after undo
        gameActive = true;      // Game is active again
        isAnimating = false;    // Ensure animation flag is off

        // Clear heatmap after undo
        moveProbabilities = Array(7).fill(0);
        if (visualizationActive) updateHeatmap(); // Only update visual if active

        console.log("Undo complete. Player's turn.");
        updateUI(); // Update UI to reflect the new state
    }

    // --- UI Update ---
    function updateUI() {
        
        const currentScorePlayer = document.getElementById('player-score');
        const currentScoreAI = document.getElementById('ai-score');
        const currentBoard = document.querySelector('.board');
        const currentUndoButton = document.getElementById('undo'); // Get current instance

        console.log("Updating UI. State:", {
            currentPlayer,
            gameActive,
            isAnimating,
            historyLength: moveHistory.length
        });

        // Highlight active player's timer/score area
        const playerTimerDiv = currentScorePlayer?.closest('.player-timer');
        const aiTimerDiv = currentScoreAI?.closest('.ai-timer');

        // Active only if it's their turn, game is active, and nothing is animating/thinking
        const isPlayerTurnActive = currentPlayer === 'player' && gameActive && !isAnimating;
        const isAiThinking = currentPlayer === 'ai' && !isAnimating && gameActive === false; // AI is 'thinking' (between turns)

        if (playerTimerDiv) playerTimerDiv.classList.toggle('active', isPlayerTurnActive);
        if (aiTimerDiv) aiTimerDiv.classList.toggle('active', currentPlayer === 'ai' && gameActive && !isAnimating); // AI area active during its turn

        // Add classes to board for visual states
        const isGameOver = !gameActive && !isAnimating;
        if (currentBoard) {
            currentBoard.classList.toggle('game-over', isGameOver);
            currentBoard.classList.toggle('thinking', isAiThinking); // Show thinking overlay via CSS
            // Disable pointer events unless it's the player's active turn
            currentBoard.style.pointerEvents = isPlayerTurnActive ? 'auto' : 'none';
        }


        // Enable/disable Undo button
        if (currentUndoButton) {
            // Disable if history empty, animating, AI is thinking, or game is over
            currentUndoButton.disabled = moveHistory.length === 0 || isAnimating || isAiThinking || isGameOver;
        }
    }


    // --- Visualization Functions (Added) ---
    function toggleVisualization() {
        visualizationActive = !visualizationActive;
        const board = document.querySelector('.board');
        const key = document.querySelector('.heatmap-key');
        const button = document.getElementById('toggle-visualization'); // Get current button instance

        board.classList.toggle('heatmap-visible', visualizationActive);
        if (key) key.style.opacity = visualizationActive ? 1 : 0;
        button.textContent = visualizationActive ? 'Hide AI Heatmap' : 'Show AI Heatmap';

        if (visualizationActive) {
            // If toggled on, immediately try to predict if it's AI's turn or show empty if player's turn
            if (currentPlayer === 'ai' && gameActive && !isAnimating) {
                predictAIMoves(); // This now calls the backend if implemented fully
            } else {
                // Clear heatmap if turned on during player's turn or inactive game
                 moveProbabilities = Array(7).fill(0);
                 updateHeatmap();
            }
        } else {
            // Clear heatmap styles if turned off
            moveProbabilities = Array(7).fill(0);
            updateHeatmap(); // This will set scores to 0, effectively clearing visual
        }
    }

    function predictAIMoves() {
        // --- Placeholder Prediction ---
        // For demonstration, keeps simulating probabilities locally.
        // A real implementation would fetch from backend's probability endpoint.
        console.log("Simulating AI move probabilities for heatmap...");
        const tempProbs = Array(7).fill(0).map((_, col) => {
            // Give higher chance to valid moves, slightly random otherwise
            return findEmptyRow(col) !== -1 ? Math.random() * 80 + 10 : Math.random() * 10;
        });
        const total = tempProbs.reduce((a, b) => a + b, 0);
        // Normalize to percentages
        moveProbabilities = total > 0 ? tempProbs.map(p => (p / total) * 100) : Array(7).fill(0);

        console.log("Predicted probabilities:", moveProbabilities);
        updateHeatmap();
        // --- End Placeholder ---

        // --- Real Implementation Idea (using /api/get_ai_move with get_probabilities=true) ---
        // // Calculate win rate for context
        // const winRate = playerStats.gamesPlayed > 0 ? playerStats.wins / playerStats.gamesPlayed : 0.5;
        // fetch('/api/get_ai_move', { // Call the main endpoint but just for probs
        //     method: 'POST',
        //     headers: { 'Content-Type': 'application/json' },
        //     body: JSON.stringify({
        //          board: boardState,
        //          algorithm: algorithmSelect.value,
        //          difficulty: difficultySelect.value,
        //          win_rate: winRate,
        //          get_probabilities: true // Explicitly ask for probabilities
        //      })
        // })
        // .then(response => {
        //      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        //      return response.json();
        // })
        // .then(data => {
        //     if (data.probabilities) {
        //         moveProbabilities = data.probabilities;
        //         updateHeatmap();
        //         console.log("Received probabilities from backend:", moveProbabilities);
        //     } else {
        //          console.warn("Backend did not return probabilities for heatmap prediction.");
        //          // Fallback to placeholder or clear heatmap
        //          moveProbabilities = Array(7).fill(0);
        //          updateHeatmap();
        //     }
        // })
        // .catch(error => {
        //     console.error("Error fetching AI probabilities:", error);
        //      // Clear heatmap on error
        //      moveProbabilities = Array(7).fill(0);
        //      updateHeatmap();
        //  });
        // --- End Real Idea ---
    }

    function updateHeatmap() {
        const cells = boardElement.querySelectorAll('.cell'); // Use boardElement context
        cells.forEach(cell => {
            const col = parseInt(cell.dataset.col);
            if (col >= 0 && col < moveProbabilities.length) {
                // Set CSS custom property; CSS handles the color gradient
                // Ensure value is between 0 and 100
                const score = Math.max(0, Math.min(100, moveProbabilities[col] || 0));
                cell.style.setProperty('--ai-score', visualizationActive ? score.toFixed(1) : 0);
            } else {
                 cell.style.removeProperty('--ai-score');
            }
        });
         // Ensure the board has the class only if active
         boardElement.classList.toggle('heatmap-visible', visualizationActive);
    }


    // --- Adaptive Difficulty Functions (Frontend Suggestion Logic Only) ---
    function updateDifficulty() {
        // This function now primarily handles *suggesting* changes.
        // Actual AI adaptation logic is backend-side using win_rate.
        const winRate = playerStats.gamesPlayed > 0 ? playerStats.wins / playerStats.gamesPlayed : 0.5;
        const currentDifficultySetting = difficultySelect.value;

        // --- Dynamic Suggestion Logic ---
        // Suggest only if enough games have been played and not currently changing difficulty
        if (playerStats.gamesPlayed > 5 && !gameActive && !isAnimating) { // Suggest only at game end
            if (winRate > 0.7 && currentDifficultySetting !== 'expert') {
                suggestDifficultyUpgrade();
            } else if (winRate < 0.3 && currentDifficultySetting !== 'easy') {
                suggestDifficultyDowngrade();
            }
        }

        // --- The code below is NO LONGER used to send parameters to the AI ---
        // --- It's kept here conceptually but doesn't affect AI behavior directly ---
        const baseSettings = difficultySettings[currentDifficultySetting];
        if (!baseSettings) {
            console.warn(`Invalid difficulty setting selected: ${currentDifficultySetting}.`);
            return { depth: 4, randomness: 0.4 }; // Fallback conceptual values
        }
        let adaptiveDepth = baseSettings.depth;
        if (winRate > 0.6) adaptiveDepth = Math.min(baseSettings.depth + 1, 10);
        if (winRate < 0.4) adaptiveDepth = Math.max(baseSettings.depth - 1, 1);
        const adaptiveRandomness = baseSettings.randomness * (1 - Math.max(0, Math.min(1, winRate * 1.2 - 0.1)));
        // console.log(`Conceptual Adaptive Settings (Not Sent): Difficulty=${currentDifficultySetting}, WinRate=${winRate.toFixed(2)}, Depth=${adaptiveDepth}, Randomness=${adaptiveRandomness.toFixed(2)}`);
        return { // Return conceptual values, primarily for potential future local use
            depth: Math.floor(adaptiveDepth),
            randomness: Math.max(0, Math.min(1, adaptiveRandomness))
        };
        // --- End conceptual calculation ---
    }


    function suggestDifficultyUpgrade() {
        // Avoid repeated suggestions - could add a flag if needed
        const currentIndex = difficultySelect.selectedIndex;
        if (currentIndex < difficultySelect.options.length - 1) {
            const nextLevel = difficultySelect.options[currentIndex + 1].value;
            // Use setTimeout to avoid blocking game flow / alert conflicts
            setTimeout(() => {
                if (confirm(`You're consistently winning! Would you like to try the '${nextLevel}' difficulty?`)) {
                    difficultySelect.value = nextLevel;
                    console.log(`Difficulty changed to ${nextLevel} based on suggestion.`);
                    // Resetting the game after difficulty change might be a good idea
                    // resetGame();
                }
            }, 500); // Delay slightly
        }
    }

    function suggestDifficultyDowngrade() {
        // Avoid repeated suggestions
        const currentIndex = difficultySelect.selectedIndex;
        if (currentIndex > 0) {
            const prevLevel = difficultySelect.options[currentIndex - 1].value;
             setTimeout(() => {
                if (confirm(`Finding it tough? Would you like to try the '${prevLevel}' difficulty?`)) {
                    difficultySelect.value = prevLevel;
                    console.log(`Difficulty changed to ${prevLevel} based on suggestion.`);
                    // resetGame();
                }
            }, 500); // Delay slightly
        }
    }

    // --Metrics
    
// Update metrics display function
function updateMetricsDisplay(metrics) {
    const metricsContainer = document.getElementById('metrics-data');
    if (!metricsContainer) return;
    
    metricsContainer.innerHTML = '';

    // Get current algorithm selection
    const currentAlgorithm = document.getElementById('algorithm').value;
    
    if (!metrics || !metrics[currentAlgorithm]) {
        showMetricsError('No metrics data available for selected algorithm');
        return;
    }

    const stats = metrics[currentAlgorithm];
    
    // Create row container
    const row = document.createElement('div');
    row.className = 'metrics-row';
    row.style.display = 'contents';

    // Algorithm Name
    const algoCell = createMetricCell(
        currentAlgorithm.charAt(0).toUpperCase() + currentAlgorithm.slice(1),
        'algorithm'
    );
    row.appendChild(algoCell);

    // Difficulty
    const diffCell = createMetricCell(
        stats.difficulty.charAt(0).toUpperCase() + stats.difficulty.slice(1),
        'difficulty'
    );
    row.appendChild(diffCell);

    // Total Moves
    const movesCell = createMetricCell(
        stats.total_moves.toString(),
        'value'
    );
    row.appendChild(movesCell);

    // Average Time
    const timeCell = createMetricCell(
        stats.avg_move_time.toFixed(2) + 's',
        'value'
    );
    row.appendChild(timeCell);

    // Win Rate
    const winRateCell = createMetricCell(
        (stats.win_rate * 100).toFixed(1) + '%',
        'value'
    );
    row.appendChild(winRateCell);

    // Performance Bar
    const perfCell = document.createElement('div');
    perfCell.className = 'metric-item';
    const perfBarContainer = document.createElement('div');
    perfBarContainer.className = 'performance-bar-container';
    const perfBar = document.createElement('div');
    perfBar.className = 'performance-bar';
    perfBar.style.width = `${stats.win_rate * 100}%`;
    perfBarContainer.appendChild(perfBar);
    perfCell.appendChild(perfBarContainer);
    row.appendChild(perfCell);

    metricsContainer.appendChild(row);
}


// Helper function to create metric cells
function createMetricCell(content, className) {
    const cell = document.createElement('div');
    cell.className = `metric-item ${className}`;
    cell.textContent = content;
    return cell;
}

// Initialize metrics when algorithm changes
document.getElementById('algorithm').addEventListener('change', loadMetrics);

// Initialize metrics when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check if metrics panel exists on this page
    if (document.querySelector('.metrics-panel')) {
        // Set up event listeners
        document.getElementById('refresh-metrics').addEventListener('click', loadMetrics);
        document.getElementById('reset-metrics').addEventListener('click', resetMetrics);
        document.getElementById('export-metrics').addEventListener('click', exportMetrics);
        document.getElementById('time-range').addEventListener('change', loadMetrics);

        // Initial load
        loadMetrics();
    }
});
// Load metrics function
function loadMetrics() {
    const refreshBtn = document.getElementById('refresh-metrics');
    if (!refreshBtn) return;
    
    refreshBtn.classList.add('loading');
    refreshBtn.disabled = true;

    // Clear existing error if any
    const errorDisplay = document.getElementById('metrics-error');
    if (errorDisplay) errorDisplay.remove();

    const timeRange = document.getElementById('time-range').value;
    const algorithm = document.getElementById('algorithm').value;
    
    fetch(`/api/metrics/current?time_range=${timeRange}&algorithm=${algorithm}`)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            if (data.status === 'success') {
                updateMetricsDisplay(data.metrics);
            } else {
                throw new Error(data.message || 'Failed to load metrics');
            }
        })
        .catch(error => {
            console.error('Error loading metrics:', error);
            showMetricsError(error.message || 'Failed to load metrics');
        })
        .finally(() => {
            refreshBtn.classList.remove('loading');
            refreshBtn.disabled = false;
        });
}

// Show error function
function showMetricsError(message) {
    const existingError = document.getElementById('metrics-error');
    if (existingError) existingError.remove();

    const errorDisplay = document.createElement('div');
    errorDisplay.id = 'metrics-error';
    errorDisplay.style.color = '#FF6F61';
    errorDisplay.style.textAlign = 'center';
    errorDisplay.style.margin = '10px 0';
    errorDisplay.style.padding = '8px';
    errorDisplay.style.backgroundColor = 'rgba(255,0,0,0.1)';
    errorDisplay.style.borderRadius = '4px';
    errorDisplay.textContent = message;

    const metricsPanel = document.querySelector('.metrics-panel');
    if (metricsPanel) {
        const metricsGrid = metricsPanel.querySelector('.metrics-grid');
        if (metricsGrid) {
            metricsPanel.insertBefore(errorDisplay, metricsGrid);
        }
    }
}

// Initialize metrics when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {    
    // Check if metrics panel exists on this page
    if (document.querySelector('.metrics-panel')) {
        // Set up event listeners
        const refreshBtn = document.getElementById('refresh-metrics');
        const resetBtn = document.getElementById('reset-metrics');
        const exportBtn = document.getElementById('export-metrics');
        const timeRange = document.getElementById('time-range');

        if (refreshBtn) refreshBtn.addEventListener('click', loadMetrics);
        if (resetBtn) resetBtn.addEventListener('click', resetMetrics);
        if (exportBtn) exportBtn.addEventListener('click', exportMetrics);
        if (timeRange) timeRange.addEventListener('change', loadMetrics);

        // Initial load
        loadMetrics();
    }
});

// Reset metrics function
function resetMetrics() {
    if (confirm('Are you sure you want to reset all metrics? This cannot be undone.')) {
        const resetBtn = document.getElementById('reset-metrics');
        resetBtn.classList.add('loading');
        resetBtn.disabled = true;

        const algorithm = document.getElementById('algorithm').value;
        
        fetch('/api/metrics/reset', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ algorithm: algorithm })
        })
        .then(response => {
            if (!response.ok) throw new Error('Reset failed');
            return response.json();
        })
        .then(data => {
            if (data.status === 'success') {
                loadMetrics();
                showMetricsSuccess('Metrics reset successfully!');
            } else {
                throw new Error(data.message || 'Failed to reset metrics');
            }
        })
        .catch(error => {
            console.error('Error resetting metrics:', error);
            showMetricsError(error.message);
        })
        .finally(() => {
            resetBtn.classList.remove('loading');
            resetBtn.disabled = false;
        });
    }
}
// Export metrics function
function exportMetrics() {
    const exportBtn = document.getElementById('export-metrics');
    exportBtn.classList.add('loading');
    exportBtn.disabled = true;

    const algorithm = document.getElementById('algorithm').value;
    
    fetch(`/api/metrics/export?algorithm=${algorithm}`)
    .then(response => {
        if (!response.ok) throw new Error('Export failed');
        return response.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `connect4_metrics_${algorithm}_${new Date().toISOString().slice(0,10)}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        showMetricsSuccess('Metrics exported successfully!');
    })
    .catch(error => {
        console.error('Export error:', error);
        showMetricsError('Failed to export metrics');
    })
    .finally(() => {
        exportBtn.classList.remove('loading');
        exportBtn.disabled = false;
    });
}
// Show success message
function showMetricsSuccess(message) {
    const successMsg = document.createElement('div');
    successMsg.style.color = '#4CAF50';
    successMsg.style.textAlign = 'center';
    successMsg.style.margin = '10px 0';
    successMsg.style.padding = '8px';
    successMsg.style.backgroundColor = 'rgba(76,175,80,0.1)';
    successMsg.style.borderRadius = '4px';
    successMsg.textContent = message;

    const metricsPanel = document.querySelector('.metrics-panel');
    if (metricsPanel) {
        const metricsGrid = metricsPanel.querySelector('.metrics-grid');
        if (metricsGrid) {
            // Remove any existing success message
            const existingSuccess = metricsPanel.querySelector('.metrics-success');
            if (existingSuccess) existingSuccess.remove();
            
            // Add new message
            successMsg.className = 'metrics-success';
            metricsPanel.insertBefore(successMsg, metricsGrid);
            
            // Auto-hide after 3 seconds
            setTimeout(() => {
                successMsg.style.opacity = '0';
                setTimeout(() => successMsg.remove(), 300);
            }, 3000);
        }
    }
}

// Update your event listeners:
document.addEventListener('DOMContentLoaded', () => {
    // Metrics controls
    const refreshBtn = document.getElementById('refresh-metrics');
    const resetBtn = document.getElementById('reset-metrics');
    const exportBtn = document.getElementById('export-metrics');
    const timeRange = document.getElementById('time-range');

    if (refreshBtn) refreshBtn.addEventListener('click', loadMetrics);
    if (resetBtn) resetBtn.addEventListener('click', resetMetrics);
    if (exportBtn) exportBtn.addEventListener('click', exportMetrics);
    if (timeRange) timeRange.addEventListener('change', loadMetrics);

    // Initial load
    loadMetrics();
});

function initGame() {
    console.log("Initializing game...");
    boardState = Array(6).fill(null).map(() => Array(7).fill(null));
    currentPlayer = 'player';
    gameActive = true;
    isAnimating = false;
    moveHistory = [];
    moveProbabilities = Array(7).fill(0);

    // Clear existing tokens
    const existingTokens = boardElement.querySelectorAll('.token');
    existingTokens.forEach(token => token.remove());

    // Reset heatmap properties
    boardElement.querySelectorAll('.cell').forEach(cell => {
        cell.style.removeProperty('--ai-score');
    });
    
    if (visualizationActive) {
        updateHeatmap(); // Clear heatmap visuals
    }

    // Recreate cell elements - this was causing issues
    // Instead, just clear the tokens and reset event listeners
    currentCells = boardElement.querySelectorAll('.cell');
    currentCells.forEach(cell => {
        // Remove any existing listeners first
        cell.replaceWith(cell.cloneNode(true));
    });

    // Get fresh references to cells after cloning
    currentCells = boardElement.querySelectorAll('.cell');
    currentCells.forEach(cell => {
        cell.addEventListener('click', () => handlePlayerMove(parseInt(cell.dataset.col)));
    });

    // Reset buttons - don't clone metrics buttons
    const resetButton = document.getElementById('reset');
    const newResetButton = resetButton.cloneNode(true);
    resetButton.parentNode.replaceChild(newResetButton, resetButton);
    newResetButton.addEventListener('click', resetGame);

    const undoButton = document.getElementById('undo');
    const newUndoButton = undoButton.cloneNode(true);
    undoButton.parentNode.replaceChild(newUndoButton, undoButton);
    newUndoButton.addEventListener('click', undoMove);

    const visualizationToggleButton = document.getElementById('toggle-visualization');
    const newVisButton = visualizationToggleButton.cloneNode(true);
    visualizationToggleButton.parentNode.replaceChild(newVisButton, visualizationToggleButton);
    newVisButton.addEventListener('click', toggleVisualization);

    // Initialize metrics buttons without cloning
    const refreshBtn = document.getElementById('refresh-metrics');
    const resetMetricsBtn = document.getElementById('reset-metrics');
    const exportBtn = document.getElementById('export-metrics');
    
    if (refreshBtn) {
        refreshBtn.onclick = loadMetrics;
    }
    if (resetMetricsBtn) {
        resetMetricsBtn.onclick = resetMetrics;
    }
    if (exportBtn) {
        exportBtn.onclick = exportMetrics;
    }

    console.log("Game Initialized.");
    updateUI();
}

    // --- Initial Setup ---
    console.log("Document loaded. Initializing game...");
    initGame(); // Start the first game
});
// --- END OF FILE main.js ---