from flask import Blueprint, request, jsonify, current_app, send_file
from flask_login import login_required, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from app.utils import get_ai_opponent, DEFAULT_API_TIMEOUT
from app.models import AIMetric, db
import logging
import traceback
import requests
import json
import numpy as np
from numbers import Number
from collections import defaultdict
from time import time
import statistics
import os
import csv
from datetime import datetime, timedelta
from io import StringIO

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "60 per hour", "15 per minute"]
)

game_bp = Blueprint('game', __name__)
logger = logging.getLogger(__name__)

# Performance metrics tracking
METRICS_LOG_FILE = 'ai_metrics.csv'
game_metrics = {
    'random': defaultdict(list),
    'minimax': defaultdict(list),
    'mcts': defaultdict(list),
    'deepseek': defaultdict(list)
}

def init_metrics_log():
    if not os.path.exists(METRICS_LOG_FILE):
        with open(METRICS_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'algorithm', 'move_time', 'move_quality', 'difficulty', 'user_id'])

init_metrics_log()

def log_metrics_to_file(algorithm, move_time, move_quality, difficulty):
    """Log metrics to CSV file"""
    try:
        with open(METRICS_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                algorithm,
                move_time,
                move_quality,
                difficulty,
                current_user.id if current_user.is_authenticated else None
            ])
    except Exception as e:
        logger.error(f"Error logging metrics to file: {str(e)}")

def record_metrics(algorithm, move_time, move_quality=None, difficulty='medium'):
    """Record performance metrics for an algorithm."""
    # Only record for the current algorithm
    if algorithm not in game_metrics:
        return
        
    metrics = game_metrics[algorithm]
    metrics['move_times'].append(move_time)
    if move_quality is not None:
        metrics['move_qualities'].append(move_quality)
    metrics['total_moves'] = len(metrics['move_times'])
    metrics['difficulty'] = difficulty
    
    # Log to database
    if current_user.is_authenticated:
        metric = AIMetric(
            algorithm=algorithm,
            move_time=move_time,
            move_quality=move_quality or 0,
            difficulty=difficulty,
            user_id=current_user.id
        )
        db.session.add(metric)
        try:
            db.session.commit()
        except Exception as e:
            logger.error(f"Error saving metric to database: {str(e)}")
            db.session.rollback()
    
    # Log to file
    log_metrics_to_file(algorithm, move_time, move_quality or 0, difficulty)
    
    # Keep only last 100 moves for performance
    for key in metrics:
        if isinstance(metrics[key], list) and len(metrics[key]) > 100:
            metrics[key] = metrics[key][-100:]

def get_algorithm_stats(algorithm):
    """Calculate statistics for an algorithm."""
    metrics = game_metrics.get(algorithm, {})
    if not metrics or not metrics.get('move_times'):
        return None
        
    stats = {
        'algorithm': algorithm,
        'total_moves': len(metrics.get('move_times', [])),
        'avg_move_time': statistics.mean(metrics['move_times']) if metrics.get('move_times') else 0,
        'min_move_time': min(metrics['move_times']) if metrics.get('move_times') else 0,
        'max_move_time': max(metrics['move_times']) if metrics.get('move_times') else 0,
        'win_rate': (sum(metrics['move_qualities']) / len(metrics['move_qualities'])) if metrics.get('move_qualities') else 0,
        'difficulty': metrics.get('difficulty', 'medium'),
        'last_updated': datetime.now().isoformat()
    }
    return stats

def validate_board(board):
    """Validate the board structure and content."""
    if not isinstance(board, list) or len(board) != 6:
        logger.warning(f"Invalid board structure: Expected 6x7 array, got {type(board)} with {len(board)} rows")
        return False
    
    valid_values = {None, 'player', 'ai'}
    
    for r_idx, row in enumerate(board):
        if not isinstance(row, list) or len(row) != 7:
            logger.warning(f"Invalid row structure at index {r_idx}: Expected list of 7, got {type(row)} with {len(row)} columns")
            return False
            
        for c_idx, cell in enumerate(row):
            if cell not in valid_values:
                logger.warning(f"Invalid cell value at [{r_idx},{c_idx}]: '{cell}'. Expected None, 'player', or 'ai'")
                return False
    
    return True

def sanitize_response_data(data):
    """Ensure all values in the response are JSON-serializable."""
    if isinstance(data, (dict, list)):
        if isinstance(data, dict):
            return {k: sanitize_response_data(v) for k, v in data.items()}
        return [sanitize_response_data(item) for item in data]
    elif isinstance(data, (str, bool)) or data is None:
        return data
    elif isinstance(data, Number):
        if np.isnan(data):
            return 0
        if np.isinf(data):
            return float('inf') if data > 0 else float('-inf')
        return float(data) if isinstance(data, (np.floating, np.integer)) else data
    else:
        return str(data)

@game_bp.route('/get_ai_move', methods=['POST'])
@login_required
@limiter.limit("10/minute")
def get_ai_move_api():
    """API Endpoint for AI move calculation with robust error handling."""
    try:
        start_time = time()
        
        if not request.is_json:
            logger.warning("Request received is not JSON")
            return jsonify({
                'error': 'Request must be JSON',
                'status': 'error',
                'code': 400
            }), 400

        data = request.get_json()
        logger.debug(f"Received AI move request data: {data}")

        required_fields = ['board', 'algorithm']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.warning(f"Missing required fields in AI move request: {missing_fields}")
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'received': list(data.keys()),
                'status': 'error',
                'code': 400
            }), 400

        board_state = data['board']
        algorithm = data['algorithm']

        if not validate_board(board_state):
            return jsonify({
                'error': 'Invalid board format or content',
                'expected': '6x7 array with null, "player", or "ai" values',
                'status': 'error',
                'code': 400
            }), 400

        difficulty = data.get('difficulty', 'medium')
        if difficulty not in ['easy', 'medium', 'hard', 'expert']:
            difficulty = 'medium'
            
        win_rate = float(data.get('win_rate', 0.5))
        win_rate = max(0.0, min(1.0, win_rate))
        
        get_probabilities = bool(data.get('get_probabilities', False))

        logger.info(f"Requesting AI move: Algo={algorithm}, Diff={difficulty}, WinRate={win_rate:.2f}")

        # Get AI instance with fallback
        try:
            ai = get_ai_opponent(
                algorithm=algorithm,
                difficulty=difficulty,
                api_key=current_app.config.get('DEEPSEEK_API_KEY'),
                win_rate=win_rate
            )
        except Exception as e:
            logger.error(f"Error creating {algorithm} AI, falling back to random: {str(e)}")
            ai = RandomAI()
            algorithm = 'random'

        # Calculate move with additional fallback
        try:
            result_data = ai.get_move(
                board_state,
                get_probabilities=get_probabilities
            )
        except Exception as e:
            logger.error(f"AI move calculation failed, falling back to random: {str(e)}")
            ai_fallback = RandomAI()
            result_data = ai_fallback.get_move(board_state, get_probabilities)
            algorithm = 'random'

        if isinstance(result_data, dict):
            response = result_data
        elif isinstance(result_data, int):
            response = {'move': result_data}
            if get_probabilities:
                logger.warning(f"Probabilities requested but not returned by {algorithm} AI")
                response['probabilities'] = [0.0] * 7
        else:
            logger.error(f"AI ({algorithm}) returned unexpected data type: {type(result_data)}")
            raise TypeError(f"AI returned unexpected format: {type(result_data)}")

        # Calculate move quality
        board_after_move = [row[:] for row in board_state]
        row = ai.get_next_open_row(board_after_move, response['move'])
        board_after_move[row][response['move']] = 'ai'
        move_quality = 1 if ai.check_win(board_after_move, 'ai', row, response['move']) else 0
        
        # Record metrics
        move_time = time() - start_time
        record_metrics(algorithm, move_time, move_quality, difficulty)

        # Add metadata to response
        response.update({
            'algorithm': algorithm,
            'difficulty_setting': difficulty,
            'status': 'success',
            'code': 200
        })

        sanitized_response = sanitize_response_data(response)
        logger.info(f"AI move response: {sanitized_response}")
        return jsonify(sanitized_response), 200

    except ValueError as e:
        logger.error(f"Value error during AI move calculation: {str(e)}", exc_info=True)
        error_msg = str(e)
        status_code = 400
        
        if "API key is missing" in error_msg:
            status_code = 400
        elif "timed out" in error_msg or "Failed to connect" in error_msg:
            status_code = 503
            error_msg = "AI service unavailable"
        
        return jsonify({
            'error': error_msg,
            'status': 'error',
            'code': status_code
        }), status_code
        
    except TypeError as e:
        logger.error(f"Type error during AI move processing: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error',
            'code': 500
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in get_ai_move_api: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'status': 'error',
            'code': 500
        }), 500

@game_bp.route('/test_deepseek', methods=['GET'])
@login_required
@limiter.limit("5/minute")
def test_deepseek():
    """API Endpoint to test Deepseek AI connectivity."""
    api_key = current_app.config.get('DEEPSEEK_API_KEY')
    if not api_key:
        logger.warning("Deepseek API key not configured.")
        return jsonify({
            'status': 'error',
            'message': 'Deepseek API key not configured on server.',
            'code': 503
        }), 503

    try:
        logger.info(f"Attempting Deepseek API test (timeout={DEFAULT_API_TIMEOUT}s)...")
        
        test_payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 5,
            "temperature": 0.1
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        api_url = "https://api.deepseek.com/v1/chat/completions"
        
        response = requests.post(
            api_url,
            headers=headers,
            json=test_payload,
            timeout=DEFAULT_API_TIMEOUT
        )
        
        response.raise_for_status()
        
        try:
            response_data = response.json()
        except ValueError as e:
            logger.error(f"Failed to parse Deepseek response: {e}")
            raise ValueError("Invalid JSON response from Deepseek API")

        if not response_data.get('choices') or not isinstance(response_data['choices'], list):
            raise ValueError("Missing or invalid 'choices' in response")
            
        if not response_data['choices'][0].get('message', {}).get('content'):
            raise ValueError("Missing message content in response")

        logger.info("Deepseek API test successful. Received response.")
        return jsonify({
            'status': 'success',
            'message': 'Deepseek AI connection successful.',
            'api_response': sanitize_response_data(response_data),
            'code': 200
        }), 200

    except requests.exceptions.Timeout:
        logger.error(f"Deepseek API test timed out after {DEFAULT_API_TIMEOUT} seconds.")
        return jsonify({
            'status': 'error',
            'message': f'Connection timed out ({DEFAULT_API_TIMEOUT}s)',
            'code': 504
        }), 504
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Deepseek API test request failed: {type(e).__name__}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Network Error: {type(e).__name__}',
            'error_details': str(e),
            'code': 502
        }), 502
        
    except ValueError as e:
        logger.error(f"Deepseek API test failed (ValueError): {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'API Response Error: {str(e)}',
            'response_data': response_data if 'response_data' in locals() else None,
            'code': 502
        }), 502
        
    except Exception as e:
        logger.error(f"Deepseek API test failed unexpectedly: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': f'Internal Server Error during test: {type(e).__name__}',
            'code': 500
        }), 500


@game_bp.route('/metrics/current', methods=['GET'])
@login_required
def get_current_metrics():
    """Endpoint to retrieve current performance metrics for specific algorithm."""
    try:
        algorithm = request.args.get('algorithm')
        if not algorithm or algorithm not in ['random', 'minimax', 'mcts', 'deepseek']:
            return jsonify({
                'status': 'error',
                'message': 'Invalid or missing algorithm parameter',
                'code': 400
            }), 400

        stats = get_algorithm_stats(algorithm)
        if not stats or stats['total_moves'] == 0:
            return jsonify({
                'status': 'error',
                'message': f'No metrics available for {algorithm}',
                'code': 404
            }), 404
        
        return jsonify({
            'status': 'success',
            'metrics': {algorithm: stats},
            'code': 200
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'code': 500
        }), 500

@game_bp.route('/metrics/history', methods=['GET'])
@login_required
def get_metrics_history():
    """Get historical metrics with filtering options."""
    try:
        # Parse query parameters
        algorithm = request.args.get('algorithm')
        time_range = request.args.get('time_range', '24h')
        limit = int(request.args.get('limit', 100))
        
        # Calculate time filter
        now = datetime.utcnow()
        if time_range == '1h':
            start_time = now - timedelta(hours=1)
        elif time_range == '24h':
            start_time = now - timedelta(days=1)
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
        elif time_range == '30d':
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(days=1)  # Default to 24h
            
        # Build query
        query = AIMetric.query.filter(
            AIMetric.timestamp >= start_time
        )
        
        if algorithm and algorithm in game_metrics:
            query = query.filter(AIMetric.algorithm == algorithm)
            
        if current_user.is_authenticated:
            query = query.filter(AIMetric.user_id == current_user.id)
            
        metrics = query.order_by(AIMetric.timestamp.desc()).limit(limit).all()
        
        return jsonify({
            'status': 'success',
            'metrics': [metric.to_dict() for metric in metrics],
            'code': 200
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving metric history: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve metric history',
            'error': str(e),
            'code': 500
        }), 500

@game_bp.route('/metrics/export', methods=['GET'])
@login_required
def export_metrics():
    """Export metrics for specific algorithm as CSV."""
    try:
        algorithm = request.args.get('algorithm')
        if not algorithm or algorithm not in game_metrics:
            return jsonify({
                'status': 'error',
                'message': 'Invalid or missing algorithm parameter',
                'code': 400
            }), 400

        # Create in-memory CSV file
        si = StringIO()
        cw = csv.writer(si)
        
        # Write header
        cw.writerow(['timestamp', 'algorithm', 'move_time', 'move_quality', 'difficulty', 'user_id'])
        
        # Get metrics for this algorithm
        query = AIMetric.query.filter_by(algorithm=algorithm)
        
        if current_user.is_authenticated:
            query = query.filter_by(user_id=current_user.id)
            
        metrics = query.order_by(AIMetric.timestamp.desc()).limit(1000).all()
        
        # Write data
        for metric in metrics:
            cw.writerow([
                metric.timestamp.isoformat(),
                metric.algorithm,
                metric.move_time,
                metric.move_quality,
                metric.difficulty,
                metric.user_id
            ])
        
        # Prepare response
        si.seek(0)
        return send_file(
            si,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{algorithm}_metrics_export_{datetime.now().date().isoformat()}.csv'
        )
        
    except Exception as e:
        logger.error(f"Error exporting metrics: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to export metrics',
            'error': str(e),
            'code': 500
        }), 500

@game_bp.route('/metrics/reset', methods=['POST'])
@login_required
def reset_metrics():
    """Reset metrics for specific algorithm."""
    try:
        data = request.get_json()
        algorithm = data.get('algorithm')
        
        if not algorithm or algorithm not in game_metrics:
            return jsonify({
                'status': 'error',
                'message': 'Invalid or missing algorithm parameter',
                'code': 400
            }), 400

        # Reset in-memory metrics
        game_metrics[algorithm] = defaultdict(list)
            
        # Clear database metrics for current user and algorithm
        if current_user.is_authenticated:
            AIMetric.query.filter_by(
                user_id=current_user.id,
                algorithm=algorithm
            ).delete()
            db.session.commit()
            
        return jsonify({
            'status': 'success',
            'message': f'Metrics for {algorithm} reset successfully',
            'code': 200
        }), 200
        
    except Exception as e:
        logger.error(f"Error resetting metrics: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to reset metrics',
            'error': str(e),
            'code': 500
        }), 500