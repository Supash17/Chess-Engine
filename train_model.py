"""
Pre-train the chess engine model and save it to disk
This script trains on 5000 positions and saves the model for fast deployment
"""

import pickle
import os
from chess_engine import PositionEvaluator, generate_training_positions

def train_and_save_model(num_positions=5000, output_file='chess_model.pkl'):
    """Train the model on positions and save to disk"""
    print(f"Training engine on {num_positions} positions...")
    
    # Create evaluator
    evaluator = PositionEvaluator()
    
    # Generate training positions
    print("Generating training positions...")
    positions = generate_training_positions(num_positions)
    
    # Train model
    print("Training Random Forest model...")
    evaluator.train(positions)
    
    # Save model to disk
    print(f"Saving model to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(evaluator, f)
    
    print(f"✅ Model trained and saved successfully!")
    print(f"   - Positions: {num_positions}")
    print(f"   - R² Score: {evaluator.r2_score:.4f}")
    print(f"   - Training MSE: {evaluator.training_mse:.6f}")
    print(f"   - File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    train_and_save_model(10000, 'chess_model.pkl')
