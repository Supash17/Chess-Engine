from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from chess_engine import ChessBoard, MinimaxEngine
import threading
import os
import pickle

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))
CORS(app)

# Initialize global variables
engine = None
opening_book = None
training_complete = False

def load_engine():
    """Load pre-trained model from disk"""
    global engine, opening_book, training_complete
    
    model_file = 'chess_model.pkl'
    
    if not os.path.exists(model_file):
        print(f"Error: {model_file} not found! Run train_model.py first.")
        training_complete = False
        return
    
    print(f"Loading pre-trained model from {model_file}...")
    try:
        with open(model_file, 'rb') as f:
            evaluator = pickle.load(f)
        engine = MinimaxEngine(evaluator, max_depth=1)
        training_complete = True
        print("✅ Model loaded successfully! Engine ready.")
    except Exception as e:
        print(f"Error loading model: {e}")
        training_complete = False

# Load engine at startup
load_engine()

@app.route('/')
@app.route('/index.html')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')

@app.route('/api/move', methods=['POST'])
def get_best_move():
    global training_complete, engine
    
    if not training_complete:
        return jsonify({'error': 'Engine still training...'}), 503
    
    if engine is None:
        return jsonify({'error': 'Engine not initialized'}), 500
    
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON data'}), 400
    
    board_fen = data.get('fen')
    print(f"Received FEN: {board_fen}")
    
    try:
        # Direct engine search with optimized move ordering
        board = ChessBoard(board_fen)
        move, evaluation = engine.find_best_move(board)
        
        response = {
            'move': str(move),
            'evaluation': float(evaluation),
            'nodes': engine.stats.nodes_evaluated,
            'cuts': engine.stats.alpha_beta_cuts
        }
        print(f"Engine move: {response['move']} | Eval: {response['evaluation']:.2f}")
        return jsonify(response)
    except Exception as e:
        print(f"Error in get_best_move: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__': 
    app.run(debug=False, port=5500, host='127.0.0.1')