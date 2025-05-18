# AI-Powered Connect Four

![Game Screenshot](/app/static/UI.png)
![Game Screenshot](/app/static/Register.png)
![Game Screenshot](/app/static/heatmap.png)

An advanced Connect Four implementation featuring multiple AI opponents with adaptive difficulty, performance metrics, and visualization tools.

## Features

- 🧠 **Multiple AI Algorithms**:
  - Minimax with Alpha-Beta Pruning
  - Monte Carlo Tree Search (MCTS)
  - Deepseek API integration
  - Random move generator (baseline)

- 📊 **Performance Analytics**:
  - Real-time metrics dashboard
  - Win rate tracking
  - Move time statistics
  - Exportable CSV reports

- 🎮 **Game Features**:
  - Adaptive difficulty scaling
  - AI move visualization (heatmap)
  - Undo/redo functionality
  - Score tracking

- 🔐 **User System**:
  - Secure authentication
  - Player profiles
  - Game history

## Technologies Used

- **Backend**: Python, Flask, SQLAlchemy
- **Frontend**: HTML5, CSS3, JavaScript
- **AI**: Minimax, MCTS, Deepseek API
- **Database**: SQLite (with Flask-SQLAlchemy)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ArtwellMa/AI-Powered-ConnectFour-Game.git
   cd AI-Powered-ConnectFour-Game
2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

4. Install dependencies:
pip install -r requirements.txt

6. Set up environment variables:
 cp .env.example .env
# Edit .env with your Deepseek API key and secret key

5. Run the application:
 python run.py
7. Open your browser to:
   127.0.0.1:5000
