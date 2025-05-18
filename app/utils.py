import random
import logging
import json
import numpy as np
import requests
from flask import current_app
from collections import defaultdict, namedtuple
import time
from functools import lru_cache
import traceback

logger = logging.getLogger(__name__)

# Define a slightly longer default timeout
DEFAULT_API_TIMEOUT = 20  # Increased from 10 to 20 seconds

# MCTS Node structure
MCTSNode = namedtuple('MCTSNode', ['state', 'parent', 'move', 'children', 'wins', 'visits', 'untried_moves'])

class BaseAI:
    def get_move(self, board_state, get_probabilities=False):
        raise NotImplementedError

    @staticmethod
    def get_next_open_row(board, col):
        for r in range(5, -1, -1):
            if board[r][col] is None:
                return r
        return -1

    @staticmethod
    def check_win(board, player, row=None, col=None):
        """Optimized win checking with optional last move coordinates"""
        if row is not None and col is not None:
            # Check only around the last move
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for dr, dc in directions:
                count = 1
                # Check positive direction
                for i in range(1, 4):
                    r, c = row + i * dr, col + i * dc
                    if 0 <= r < 6 and 0 <= c < 7 and board[r][c] == player:
                        count += 1
                    else:
                        break
                # Check negative direction
                for i in range(1, 4):
                    r, c = row - i * dr, col - i * dc
                    if 0 <= r < 6 and 0 <= c < 7 and board[r][c] == player:
                        count += 1
                    else:
                        break
                if count >= 4:
                    return True
            return False
        else:
            # Full board check (slower)
            # Check horizontal
            for r in range(6):
                for c in range(4):
                    if all(board[r][c+i] == player for i in range(4)):
                        return True
            # Check vertical
            for c in range(7):
                for r in range(3):
                    if all(board[r+i][c] == player for i in range(4)):
                        return True
            # Check positive diagonal
            for r in range(3):
                for c in range(4):
                    if all(board[r+i][c+i] == player for i in range(4)):
                        return True
            # Check negative diagonal
            for r in range(3):
                for c in range(4):
                    if all(board[r+3-i][c+i] == player for i in range(4)):
                        return True
            return False

class RandomAI(BaseAI):
    def get_move(self, board_state, get_probabilities=False):
        valid_moves = [col for col in range(7) if board_state[0][col] is None]

        if not valid_moves:
            move = -1
        else:
            move = random.choice(valid_moves)

        if get_probabilities:
            if not valid_moves:
                probs = [0.0] * 7
            else:
                # More realistic random probabilities (sum to 100)
                raw_probs = np.random.dirichlet(np.ones(len(valid_moves)), size=1)[0]
                probs_dict = {move: p for move, p in zip(valid_moves, raw_probs)}
                probs = [round(probs_dict.get(c, 0) * 100, 1) for c in range(7)]

            return {
                'move': move,
                'probabilities': probs
            }
        return move

class MinimaxAI(BaseAI):
    def __init__(self, depth=4, randomness=0.2, use_heuristics=True, use_alpha_beta=True):
        self.depth = depth
        self.randomness = randomness
        self.use_heuristics = use_heuristics
        self.use_alpha_beta = use_alpha_beta
        self.transposition_table = {}
        self.move_order = [3, 2, 4, 1, 5, 0, 6]  # Center columns first

    def evaluate_position(self, board_tuple, player):
        """Memoized evaluation function that takes a tuple"""
        board = [list(row) for row in board_tuple]  # Convert back to list of lists
        opponent = 'ai' if player == 'player' else 'player'
        score = 0
        
        # Center column preference
        center_array = [board[r][3] for r in range(6)]
        center_count = center_array.count(player)
        score += center_count * 6

        # Evaluate all possible 4-in-a-row windows
        for r in range(6):
            for c in range(4):
                # Horizontal
                window = [board[r][c+i] for i in range(4)]
                score += self.evaluate_window(window, player)
                
        for c in range(7):
            for r in range(3):
                # Vertical
                window = [board[r+i][c] for i in range(4)]
                score += self.evaluate_window(window, player)
                
        for r in range(3):
            for c in range(4):
                # Positive diagonal
                window = [board[r+i][c+i] for i in range(4)]
                score += self.evaluate_window(window, player)
                
                # Negative diagonal
                window = [board[r+3-i][c+i] for i in range(4)]
                score += self.evaluate_window(window, player)
                
        return score

    def evaluate_window(self, window, player):
        score = 0
        opponent = 'ai' if player == 'player' else 'player'
        
        if window.count(player) == 4:
            score += 10000
        elif window.count(player) == 3 and window.count(None) == 1:
            score += 10
        elif window.count(player) == 2 and window.count(None) == 2:
            score += 3

        if window.count(opponent) == 4:
            score -= 100000
        elif window.count(opponent) == 3 and window.count(None) == 1:
            score -= 80
        elif window.count(opponent) == 2 and window.count(None) == 2:
            score -= 5
            
        return score

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        board_tuple = tuple(tuple(row) for row in board)
        
        # Check transposition table
        if board_tuple in self.transposition_table:
            return self.transposition_table[board_tuple]
            
        valid_moves = [col for col in self.move_order if board[0][col] is None]
        
        # Terminal node check
        if self.check_win(board, 'ai'):
            return (None, 100000000 + depth)  # Prefer faster wins
        elif self.check_win(board, 'player'):
            return (None, -100000000 - depth)  # Prefer slower losses
        elif depth == 0 or not valid_moves:
            return (None, self.evaluate_position(board_tuple, 'ai'))

        if maximizing_player:
            value = -float('inf')
            column = valid_moves[0]
            
            for col in valid_moves:
                row = self.get_next_open_row(board, col)
                b_copy = [row[:] for row in board]
                b_copy[row][col] = 'ai'
                
                new_score = self.minimax(b_copy, depth-1, alpha, beta, False)[1]
                
                if new_score > value:
                    value = new_score
                    column = col
                    
                if self.use_alpha_beta:
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                        
            result = (column, value)
        else:
            value = float('inf')
            column = valid_moves[0]
            
            for col in valid_moves:
                row = self.get_next_open_row(board, col)
                b_copy = [row[:] for row in board]
                b_copy[row][col] = 'player'
                
                new_score = self.minimax(b_copy, depth-1, alpha, beta, True)[1]
                
                if new_score < value:
                    value = new_score
                    column = col
                    
                if self.use_alpha_beta:
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
                        
            result = (column, value)
            
        # Store in transposition table
        self.transposition_table[board_tuple] = result
        return result

    def get_move(self, board_state, get_probabilities=False):
        valid_moves = [col for col in range(7) if board_state[0][col] is None]
        if not valid_moves:
            return {'move': -1, 'probabilities': [0.0] * 7} if get_probabilities else -1

        # Immediate win/loss check
        for col in valid_moves:
            row = self.get_next_open_row(board_state, col)
            board_state[row][col] = 'ai'
            if self.check_win(board_state, 'ai', row, col):
                board_state[row][col] = None
                probs = [100.0 if c == col else 0.0 for c in range(7)]
                return {'move': col, 'probabilities': probs} if get_probabilities else col
            board_state[row][col] = None

            board_state[row][col] = 'player'
            if self.check_win(board_state, 'player', row, col):
                board_state[row][col] = None
                probs = [100.0 if c == col else 0.0 for c in range(7)]
                return {'move': col, 'probabilities': probs} if get_probabilities else col
            board_state[row][col] = None

        # Apply randomness
        if random.random() < self.randomness:
            move = random.choice(valid_moves)
            if get_probabilities:
                prob_per_move = round(100.0 / len(valid_moves), 1)
                probs = [prob_per_move if c in valid_moves else 0.0 for c in range(7)]
                if valid_moves: 
                    probs[valid_moves[-1]] = round(100.0 - sum(probs[:-1]), 1)
                return {'move': move, 'probabilities': probs}
            return move

        # Run Minimax
        col, score = self.minimax(board_state, self.depth, -float('inf'), float('inf'), True)

        if get_probabilities:
            # Estimate probabilities based on move ordering
            probs = [0.0] * 7
            probs[col] = 100.0
            return {'move': col, 'probabilities': probs}
        return col

class MCTSAI(BaseAI):
    def __init__(self, iterations=1000, exploration_weight=1.41, time_limit=None):
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.time_limit = time_limit  # seconds
        self.nodes_expanded = 0

    def get_move(self, board_state, get_probabilities=False):
        valid_moves = [col for col in range(7) if board_state[0][col] is None]
        if not valid_moves:
            return {'move': -1, 'probabilities': [0.0] * 7} if get_probabilities else -1

        # Immediate win/loss check
        for col in valid_moves:
            row = self.get_next_open_row(board_state, col)
            board_state[row][col] = 'ai'
            if self.check_win(board_state, 'ai', row, col):
                board_state[row][col] = None
                probs = [100.0 if c == col else 0.0 for c in range(7)]
                return {'move': col, 'probabilities': probs} if get_probabilities else col
            board_state[row][col] = None

            board_state[row][col] = 'player'
            if self.check_win(board_state, 'player', row, col):
                board_state[row][col] = None
                probs = [100.0 if c == col else 0.0 for c in range(7)]
                return {'move': col, 'probabilities': probs} if get_probabilities else col
            board_state[row][col] = None

        root = self.Node(board_state)
        start_time = time.time()
        iterations = 0

        while (iterations < self.iterations and 
               (self.time_limit is None or time.time() - start_time < self.time_limit)):
            node = root
            board_copy = [row[:] for row in board_state]

            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.select_child(self.exploration_weight)
                row = self.get_next_open_row(board_copy, node.move)
                board_copy[row][node.move] = 'ai' if node.player_to_move == 'ai' else 'player'

            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                move = node.select_untried_move()
                row = self.get_next_open_row(board_copy, move)
                board_copy[row][move] = node.player_to_move
                node = node.add_child(move, board_copy)
                self.nodes_expanded += 1

            # Simulation
            result = self.simulate_game(board_copy, node.player_to_move)

            # Backpropagation
            while node is not None:
                node.update(result)
                node = node.parent
            iterations += 1

        logger.debug(f"MCTS completed {iterations} iterations in {time.time()-start_time:.2f}s")
        
        # Get best move
        best_move = root.best_child(0).move
        
        if get_probabilities:
            total_visits = sum(child.visits for child in root.children)
            probabilities = [0.0] * 7
            for child in root.children:
                probabilities[child.move] = (child.visits / total_visits) * 100
            return {'move': best_move, 'probabilities': probabilities}
            
        return best_move

    def simulate_game(self, board, player):
        current_player = player
        while True:
            valid_moves = [col for col in range(7) if board[0][col] is None]
            if not valid_moves:
                return 0  # Draw
                
            move = random.choice(valid_moves)
            row = self.get_next_open_row(board, move)
            board[row][move] = current_player
            
            if self.check_win(board, current_player, row, move):
                return 1 if current_player == 'ai' else -1
                
            current_player = 'player' if current_player == 'ai' else 'ai'

    class Node:
        def __init__(self, board_state, parent=None, move=None, player_to_move='ai'):
            self.board_state = [row[:] for row in board_state]
            self.parent = parent
            self.move = move
            self.player_to_move = player_to_move
            self.children = []
            self.wins = 0
            self.visits = 0
            self.untried_moves = [col for col in range(7) if board_state[0][col] is None]

        def is_terminal(self):
            return (BaseAI.check_win(self.board_state, 'ai') or 
                    BaseAI.check_win(self.board_state, 'player') or 
                    all(self.board_state[0][col] is not None for col in range(7)))

        def is_fully_expanded(self):
            return len(self.untried_moves) == 0

        def select_untried_move(self):
            return random.choice(self.untried_moves)

        def add_child(self, move, board_state):
            child = MCTSAI.Node(
                board_state,
                parent=self,
                move=move,
                player_to_move='player' if self.player_to_move == 'ai' else 'ai'
            )
            self.untried_moves.remove(move)
            self.children.append(child)
            return child

        def update(self, result):
            self.visits += 1
            self.wins += result

        def select_child(self, exploration_weight):
            log_total_visits = np.log(self.visits)
            
            def uct_score(child):
                exploit = child.wins / child.visits
                explore = exploration_weight * np.sqrt(log_total_visits / child.visits)
                return exploit + explore
                
            return max(self.children, key=uct_score)

        def best_child(self, exploration_weight=0):
            if not self.children:
                return None
                
            if exploration_weight == 0:
                return max(self.children, key=lambda c: c.visits)
            else:
                return self.select_child(exploration_weight)

class DeepseekAI(BaseAI):
    def __init__(self, api_key):
        if not api_key:
            logger.error("DeepseekAI initialized without API key.")
            raise ValueError("Deepseek API key is required but was not provided.")
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.cache = {}

    def _format_board(self, board_state):
        # Convert board to numerical representation for the prompt
        numerical_board = []
        for row in board_state:
            numerical_row = [0 if cell is None else (1 if cell == 'player' else 2) for cell in row]
            numerical_board.append(numerical_row)
        # Format as a simple string grid
        return "\n".join([" ".join(map(str, row)) for row in numerical_board])

    def _parse_response(self, response_text, valid_moves):
        try:
            # Try parsing as JSON
            try:
                data = json.loads(response_text)
                if isinstance(data, dict) and 'move' in data and isinstance(data['move'], int):
                    move = data['move']
                    if move in valid_moves:
                        return move
            except (json.JSONDecodeError, TypeError):
                pass

            # Try finding the first valid number in the string
            import re
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                move = int(numbers[0])
                if move in valid_moves:
                    return move

            return None
        except Exception as e:
            logger.error(f"Error parsing Deepseek response: {e}")
            return None

    def get_move(self, board_state, get_probabilities=False):
        valid_moves = [col for col in range(7) if board_state[0][col] is None]
        if not valid_moves:
            return {'move': -1, 'probabilities': [0.0] * 7} if get_probabilities else -1

        # Check cache
        board_tuple = tuple(tuple(row) for row in board_state)
        if board_tuple in self.cache:
            return self.cache[board_tuple]

        # Immediate win/loss check
        for col in valid_moves:
            row = self.get_next_open_row(board_state, col)
            board_state[row][col] = 'ai'
            if self.check_win(board_state, 'ai', row, col):
                board_state[row][col] = None
                probs = [100.0 if c == col else 0.0 for c in range(7)]
                result = {'move': col, 'probabilities': probs} if get_probabilities else col
                self.cache[board_tuple] = result
                return result
            board_state[row][col] = None

            board_state[row][col] = 'player'
            if self.check_win(board_state, 'player', row, col):
                board_state[row][col] = None
                probs = [100.0 if c == col else 0.0 for c in range(7)]
                result = {'move': col, 'probabilities': probs} if get_probabilities else col
                self.cache[board_tuple] = result
                return result
            board_state[row][col] = None

        formatted_board = self._format_board(board_state)
        prompt = f"""You are playing Connect Four as AI (player 2, yellow). The board state is below (0=empty, 1=player/red, 2=AI/yellow). Rows are 0 (top) to 5 (bottom), columns 0 (left) to 6 (right).

Board:
{formatted_board}

Your goal is to win or block the opponent (player 1).
The available columns to play are: {valid_moves}.

Choose the best column number from the valid moves. Respond with ONLY the chosen column number (e.g., "3"). Do not include any other text, explanation, or formatting.
"""

        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 10,
            "stream": False
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=DEFAULT_API_TIMEOUT
            )
            response.raise_for_status()

            response_data = response.json()
            move_text = response_data['choices'][0]['message']['content'].strip()
            move = self._parse_response(move_text, valid_moves)

            if move is None:
                move = random.choice(valid_moves)
                probs = [round(100.0/len(valid_moves),1) if c in valid_moves else 0.0 for c in range(7)]
                if valid_moves: 
                    probs[valid_moves[-1]] = round(100.0 - sum(probs[:-1]), 1)
            else:
                probs = [100.0 if c == move else 0.0 for c in range(7)]

            result = {'move': move, 'probabilities': probs} if get_probabilities else move
            self.cache[board_tuple] = result
            return result

        except requests.exceptions.Timeout:
            logger.error(f"Deepseek API request timed out after {DEFAULT_API_TIMEOUT} seconds.")
            raise ValueError(f"AI service timed out ({DEFAULT_API_TIMEOUT}s).")
        except requests.exceptions.RequestException as e:
            logger.error(f"Deepseek API request failed: {str(e)}\n{traceback.format_exc()}", exc_info=False)
            raise ValueError(f"Failed to connect to AI service: {type(e).__name__}")
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error processing Deepseek response: {str(e)}\n{traceback.format_exc()}", exc_info=False)
            raise ValueError(f"Invalid response from AI: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during Deepseek move generation: {str(e)}\n{traceback.format_exc()}", exc_info=False)
            raise ValueError("Unknown AI service error.")

def get_ai_opponent(algorithm, difficulty='medium', api_key=None, win_rate=0.5):
    logger.info(f"Creating AI opponent: algorithm='{algorithm}', difficulty='{difficulty}', win_rate={win_rate:.2f}")
    
    # Base parameters by difficulty
    difficulty_params = {
        'easy': {
            'random': {'randomness': 0.8},
            'minimax': {'depth': 2, 'randomness': 0.5, 'use_heuristics': True, 'use_alpha_beta': True},
            'mcts': {'iterations': 500, 'exploration_weight': 1.5, 'time_limit': 2},
            'deepseek': {}  # Empty config for deepseek
        },
        'medium': {
            'random': {'randomness': 0.5},
            'minimax': {'depth': 4, 'randomness': 0.2, 'use_heuristics': True, 'use_alpha_beta': True},
            'mcts': {'iterations': 1000, 'exploration_weight': 1.41, 'time_limit': 3},
            'deepseek': {}  # Empty config for deepseek
        },
        'hard': {
            'random': {'randomness': 0.2},
            'minimax': {'depth': 5, 'randomness': 0.1, 'use_heuristics': True, 'use_alpha_beta': True},
            'mcts': {'iterations': 2000, 'exploration_weight': 1.3, 'time_limit': 5},
            'deepseek': {}  # Empty config for deepseek
        },
        'expert': {
            'random': {'randomness': 0},
            'minimax': {'depth': 6, 'randomness': 0, 'use_heuristics': True, 'use_alpha_beta': True},
            'mcts': {'iterations': 4000, 'exploration_weight': 1.2, 'time_limit': 8},
            'deepseek': {}  # Empty config for deepseek
        }
    }

    if difficulty not in difficulty_params:
        logger.warning(f"Invalid difficulty '{difficulty}'. Defaulting to medium.")
        difficulty = 'medium'
    
    if algorithm not in difficulty_params[difficulty]:
        logger.warning(f"Invalid algorithm '{algorithm}' for difficulty '{difficulty}'. Defaulting to minimax.")
        algorithm = 'minimax'

    params = difficulty_params[difficulty][algorithm].copy()
    
    # Adaptive adjustments based on player performance
    if algorithm == 'minimax':
        # Scale depth based on win rate
        depth_adjustment = 0
        if win_rate > 0.7: depth_adjustment = 1
        elif win_rate < 0.3: depth_adjustment = -1
        params['depth'] = max(1, min(8, params['depth'] + depth_adjustment))
        
        # Scale randomness inversely with win rate
        params['randomness'] *= (1 - min(1, max(0, win_rate * 1.5 - 0.25)))
    
    elif algorithm == 'mcts':
        # Scale iterations based on performance
        iteration_adjustment = 0
        if win_rate > 0.7: iteration_adjustment = int(params['iterations'] * 0.5)
        elif win_rate < 0.3: iteration_adjustment = -int(params['iterations'] * 0.25)
        params['iterations'] = max(100, params['iterations'] + iteration_adjustment)
        
        # Reduce exploration as player gets better
        params['exploration_weight'] *= (1 - min(0.5, max(0, win_rate - 0.5)))

    # Create AI instance
    if algorithm == 'random':
        return RandomAI(**params)
    elif algorithm == 'minimax':
        return MinimaxAI(**params)
    elif algorithm == 'mcts':
        return MCTSAI(**params)
    elif algorithm == 'deepseek':
        if not api_key:
            logger.error("Attempted to create DeepseekAI without providing an API key.")
            raise ValueError("Deepseek algorithm selected, but API key is missing in configuration.")
        return DeepseekAI(api_key=api_key)
    else:
        logger.error(f"Unknown AI algorithm requested: '{algorithm}'. Falling back to Random.")
        return RandomAI()