"""
Generate pre-computed opening book with depth=5 search
This will take time but creates instant move lookups
"""

import pickle
import os
from chess_engine import PositionEvaluator, MinimaxEngine, ChessBoard, generate_training_positions
import chess

def generate_opening_book(num_positions=50000, max_depth=5, output_file='opening_book.pkl'):
    """
    Generate opening book by pre-computing best moves for positions
    
    Args:
        num_positions: Number of positions to analyze
        max_depth: Depth for minimax search (higher = better but slower)
        output_file: File to save the opening book
    """
    print(f"Generating opening book with {num_positions} positions...")
    print(f"Search depth: {max_depth}")
    print(f"This may take 30-60 minutes...\n")
    
    # Load pre-trained model
    print("Loading pre-trained model...")
    try:
        with open('chess_model.pkl', 'rb') as f:
            evaluator = pickle.load(f)
        print("✅ Model loaded!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create engine with depth=5
    engine = MinimaxEngine(evaluator, max_depth=max_depth)
    
    # Generate positions
    print("Generating training positions...")
    positions = generate_training_positions(num_positions)
    print(f"✅ Generated {len(positions)} positions!\n")
    
    # Create opening book dictionary
    opening_book = {}
    
    print(f"Computing best moves for {len(positions)} positions...")
    print("(This will take a while - saving progress every 500 positions)\n")
    
    for idx, (fen, _) in enumerate(positions):
        progress = idx + 1
        
        if progress % 500 == 0:
            print(f"Progress: {progress}/{len(positions)} | Book size: {len(opening_book)}")
            # Save progress every 500 positions
            with open(output_file, 'wb') as f:
                pickle.dump(opening_book, f)
            print(f"   ✅ Auto-saved to {output_file}\n")
        
        try:
            board = ChessBoard(fen)
            best_move, evaluation = engine.find_best_move(board)
            
            if best_move:
                # Store move and evaluation
                opening_book[fen] = {
                    'move': str(best_move),
                    'evaluation': float(evaluation),
                    'nodes': engine.stats.nodes_evaluated,
                    'depth': max_depth
                }
        except Exception as e:
            # Skip problematic positions
            pass
    
    # Final save
    print(f"\n✅ Saving final opening book with {len(opening_book)} positions...")
    with open(output_file, 'wb') as f:
        pickle.dump(opening_book, f)
    
    file_size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"✅ Opening book generated successfully!")
    print(f"   - Total positions: {len(opening_book)}")
    print(f"   - File size: {file_size_mb:.2f} MB")
    print(f"   - Lookup time: O(1) at runtime!")
    print(f"\nYou can now run: python app.py")
    
    return opening_book

if __name__ == '__main__':
    # Generate book with 50k positions and depth=5
    # This will take ~30-60 minutes but creates instant lookups!
    generate_opening_book(num_positions=50000, max_depth=5, output_file='opening_book.pkl')

