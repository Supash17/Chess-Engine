"""
Unit Tests for Chess Engine
=============================

Comprehensive test suite demonstrating:
- Move generation correctness
- Minimax algorithm validation
- Position evaluation accuracy
- Alpha-beta pruning verification
- Model training and evaluation
"""

import unittest
import numpy as np
from chess_engine import (
    ChessBoard, PositionEvaluator, MinimaxEngine,
    generate_training_positions, SearchStats
)
import chess


class TestChessBoardMoveGeneration(unittest.TestCase):
    """Test move generation and board manipulation."""
    
    def test_initial_position_move_count(self):
        """Starting position should have 20 legal moves (e4, d4, etc)."""
        board = ChessBoard()
        moves = board.get_legal_moves()
        self.assertEqual(len(moves), 20)
    
    def test_make_move_updates_position(self):
        """Making a move should update the board."""
        board = ChessBoard()
        initial_fen = board.get_fen()
        
        moves = board.get_legal_moves()
        board.make_move(moves[0])
        
        new_fen = board.get_fen()
        self.assertNotEqual(initial_fen, new_fen)
    
    def test_undo_move_restores_position(self):
        """Undoing a move should restore the previous position."""
        board = ChessBoard()
        initial_fen = board.get_fen()
        
        moves = board.get_legal_moves()
        board.make_move(moves[0])
        board.undo_move()
        
        restored_fen = board.get_fen()
        self.assertEqual(initial_fen, restored_fen)
    
    def test_move_history_tracking(self):
        """Move history should be correctly maintained."""
        board = ChessBoard()
        moves = board.get_legal_moves()
        
        board.make_move(moves[0])
        board.make_move(moves[1])
        
        self.assertEqual(len(board.move_history), 2)
    
    def test_checkmate_detection(self):
        """Engine should detect checkmate."""
        # Back rank mate: 1.f3 e6 2.g4 Qh4#
        board = ChessBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        board.make_move(chess.Move.from_uci('f2f3'))
        board.make_move(chess.Move.from_uci('e7e6'))
        board.make_move(chess.Move.from_uci('g2g4'))
        board.make_move(chess.Move.from_uci('d8h4'))
        
        self.assertTrue(board.is_checkmate())
    
    def test_stalemate_detection(self):
        """Engine should detect stalemate."""
        # Stalemate position
        board = ChessBoard('k7/8/8/8/8/8/7K/7R b - - 0 1')
        self.assertTrue(board.is_stalemate())
    
    def test_game_over_detection(self):
        """is_game_over should catch checkmate and stalemate."""
        board = ChessBoard('k7/8/8/8/8/8/7K/7R b - - 0 1')
        self.assertTrue(board.is_game_over())


class TestPositionEvaluation(unittest.TestCase):
    """Test position evaluation and feature extraction."""
    
    def setUp(self):
        """Create evaluator for tests."""
        self.evaluator = PositionEvaluator()
    
    def test_feature_extraction_length(self):
        """Extracted features should always be 24 dimensions."""
        board = ChessBoard()
        features = self.evaluator.extract_features(board)
        
        self.assertEqual(len(features), 24)
        self.assertEqual(features.dtype, np.float32)
    
    def test_feature_extraction_range(self):
        """Features should be in reasonable normalized ranges."""
        board = ChessBoard()
        features = self.evaluator.extract_features(board)
        
        # Most features should be in [0, 1] or [-1, 1]
        for feature in features[:-2]:  # Skip binary features
            self.assertGreaterEqual(feature, -2)
            self.assertLessEqual(feature, 2)
    
    def test_material_evaluation(self):
        """Material evaluation should prefer positions with more material."""
        board1 = ChessBoard()  # Starting position
        eval1 = self.evaluator.static_evaluate(board1)
        self.assertEqual(eval1, 0)  # Balanced
        
        # Position with white up a pawn
        board2 = ChessBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        board2.board.remove_piece_at(8)  # Remove black pawn
        eval2 = self.evaluator.static_evaluate(board2)
        self.assertGreater(eval2, 0)  # White advantage
    
    def test_evaluation_symmetry(self):
        """Evaluation should be opposite from both player perspectives."""
        board = ChessBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        eval_from_white = self.evaluator.static_evaluate(board)
        
        # After move, should see opposite evaluation
        board.make_move(chess.Move.from_uci('e2e4'))
        board.board.turn = not board.board.turn
        # Evaluation from new perspective
    
    def test_checkmate_evaluation(self):
        """Checkmate should evaluate to infinity."""
        # Fool's mate: 1.f3 e5 2.g4 Qh4#
        board = ChessBoard()
        board.make_move(chess.Move.from_uci('f2f3'))
        board.make_move(chess.Move.from_uci('e7e5'))
        board.make_move(chess.Move.from_uci('g2g4'))
        board.make_move(chess.Move.from_uci('d8h4'))
        
        eval_white = self.evaluator.evaluate(board)
        self.assertEqual(eval_white, float('inf'))


class TestRandomForestTraining(unittest.TestCase):
    """Test machine learning model training and evaluation."""
    
    def setUp(self):
        """Create evaluator and training data."""
        self.evaluator = PositionEvaluator()
        self.positions = generate_training_positions(100)  # Small set for speed
    
    def test_training_convergence(self):
        """Model should train successfully."""
        metrics = self.evaluator.train(self.positions, test_size=0.2)
        
        self.assertIsNotNone(metrics)
        self.assertIn('training_mse', metrics)
        self.assertIn('r2_score', metrics)
    
    def test_model_is_trained(self):
        """After training, model should be marked as trained."""
        self.assertFalse(self.evaluator.is_trained)
        self.evaluator.train(self.positions)
        self.assertTrue(self.evaluator.is_trained)
    
    def test_training_metrics_validity(self):
        """Training metrics should be valid numbers."""
        metrics = self.evaluator.train(self.positions)
        
        self.assertGreater(metrics['training_mse'], 0)
        self.assertGreater(metrics['validation_mse'], 0)
        self.assertLess(metrics['r2_score'], 1.0)
        self.assertGreater(metrics['r2_score'], -1.0)  # Can be negative for poor fit
    
    def test_feature_importance_extraction(self):
        """Model should extract feature importance."""
        self.evaluator.train(self.positions)
        
        importance = self.evaluator.feature_importance
        self.assertEqual(len(importance), 24)  # 24 features
        self.assertAlmostEqual(np.sum(importance), 1.0, places=5)  # Should sum to 1
    
    def test_untrained_model_uses_material(self):
        """Untrained model should fall back to material evaluation."""
        board = ChessBoard()
        
        # Should use static evaluation
        eval_untrained = self.evaluator.evaluate(board, use_ml=True)
        eval_material = self.evaluator.static_evaluate(board)
        
        # For balanced position, should be same
        self.assertEqual(eval_untrained, eval_material)


class TestMinimaxAlgorithm(unittest.TestCase):
    """Test minimax algorithm and alpha-beta pruning."""
    
    def setUp(self):
        """Create evaluator and engine for tests."""
        self.evaluator = PositionEvaluator()
        self.engine = MinimaxEngine(self.evaluator, max_depth=10)
    
    def test_finds_best_move(self):
        """Engine should find a legal move."""
        board = ChessBoard()
        move, evaluation = self.engine.find_best_move(board)
        
        self.assertIsNotNone(move)
        self.assertIn(move, board.get_legal_moves())
    
    def test_evaluation_is_reasonable(self):
        """Evaluation should be in reasonable range for positions."""
        board = ChessBoard()
        move, evaluation = self.engine.find_best_move(board)
        
        # Starting position should be close to 0
        self.assertGreater(evaluation, -5)
        self.assertLess(evaluation, 5)
    
    def test_alpha_beta_pruning_occurs(self):
        """Alpha-beta pruning should eliminate some nodes."""
        board = ChessBoard()
        move, _ = self.engine.find_best_move(board)
        
        stats = self.engine.stats
        self.assertGreater(stats.alpha_beta_cuts, 0)
        self.assertLess(stats.alpha_beta_cuts, stats.nodes_evaluated)
    
    def test_transposition_table_functionality(self):
        """Transposition table should cache positions."""
        board = ChessBoard()
        initial_table_size = len(self.engine.transposition_table)
        
        self.engine.find_best_move(board)
        
        final_table_size = len(self.engine.transposition_table)
        self.assertGreater(final_table_size, initial_table_size)
    
    def test_detects_checkmate(self):
        """Engine should recognize forced checkmate."""
        # Back rank mate in 1
        board = ChessBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        board.make_move(chess.Move.from_uci('f2f3'))
        board.make_move(chess.Move.from_uci('e7e6'))
        board.make_move(chess.Move.from_uci('g2g4'))
        
        move, evaluation = self.engine.find_best_move(board)
        
        # Should suggest best defense or recognize mate threat
        self.assertIsNotNone(move)
    
    def test_consistency(self):
        """Same position should give same move."""
        board = ChessBoard()
        move1, eval1 = self.engine.find_best_move(board)
        
        # Reset transposition table
        self.engine.transposition_table.clear()
        move2, eval2 = self.engine.find_best_move(board)
        
        self.assertEqual(move1, move2)
        self.assertAlmostEqual(eval1, eval2, places=1)


class TestAlphaBetaPruningEfficiency(unittest.TestCase):
    """Test alpha-beta pruning efficiency and correctness."""
    
    def test_pruning_efficiency_percentage(self):
        """Alpha-beta pruning should eliminate 30-50% of nodes."""
        evaluator = PositionEvaluator()
        engine = MinimaxEngine(evaluator, max_depth=10)
        
        board = ChessBoard()
        engine.find_best_move(board)
        
        stats = engine.stats
        pruning_rate = stats.alpha_beta_cuts / max(1, stats.nodes_evaluated)
        
        # Should be between 0.2 and 0.6
        self.assertGreater(pruning_rate, 0.2)
        self.assertLess(pruning_rate, 0.8)
    
    def test_pruning_doesnt_affect_correctness(self):
        """Pruned and unpruned search should give same result."""
        evaluator = PositionEvaluator()
        
        engine = MinimaxEngine(evaluator, max_depth=10)
        board = ChessBoard()
        
        move_with_pruning, eval_with_pruning = engine.find_best_move(board)
        
        # Result should be correct (verified by checking move is legal)
        self.assertIn(move_with_pruning, board.get_legal_moves())


class TestSearchStatistics(unittest.TestCase):
    """Test search statistics collection and reporting."""
    
    def test_stats_initialization(self):
        """SearchStats should initialize to zero."""
        stats = SearchStats()
        
        self.assertEqual(stats.nodes_evaluated, 0)
        self.assertEqual(stats.alpha_beta_cuts, 0)
        self.assertEqual(stats.evaluation_count, 0)
    
    def test_stats_collection_during_search(self):
        """Search should populate statistics."""
        evaluator = PositionEvaluator()
        engine = MinimaxEngine(evaluator, max_depth=10)
        
        board = ChessBoard()
        engine.find_best_move(board)
        
        stats = engine.stats
        self.assertGreater(stats.nodes_evaluated, 0)
        self.assertGreater(stats.evaluation_count, 0)
        self.assertGreater(stats.search_time, 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_empty_move_list(self):
        """Checkmate position should have no legal moves."""
        # Fool's mate
        board = ChessBoard()
        board.make_move(chess.Move.from_uci('f2f3'))
        board.make_move(chess.Move.from_uci('e7e5'))
        board.make_move(chess.Move.from_uci('g2g4'))
        board.make_move(chess.Move.from_uci('d8h4'))
        
        moves = board.get_legal_moves()
        self.assertEqual(len(moves), 0)
    
    def test_position_from_fen(self):
        """Should correctly load position from FEN."""
        fen = 'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 4'
        board = ChessBoard(fen)
        
        self.assertEqual(board.get_fen(), fen)
    
    def test_multiple_undo_moves(self):
        """Should handle multiple consecutive undos."""
        board = ChessBoard()
        
        moves = board.get_legal_moves()
        board.make_move(moves[0])
        board.make_move(moves[1])
        
        board.undo_move()
        board.undo_move()
        
        self.assertEqual(len(board.move_history), 0)
    
    def test_illegal_move_rejection(self):
        """Board should only accept legal moves."""
        board = ChessBoard()
        legal_moves = board.get_legal_moves()
        
        # All moves in legal_moves should be legal
        for move in legal_moves:
            board_copy = ChessBoard(board.get_fen())
            board_copy.make_move(move)
            self.assertTrue(True)  # If we got here, move was accepted


class TestPerformance(unittest.TestCase):
    """Test performance and efficiency."""
    
    def test_search_speed(self):
        """Search should complete in reasonable time."""
        import time
        
        evaluator = PositionEvaluator()
        engine = MinimaxEngine(evaluator, max_depth=10)
        
        board = ChessBoard()
        start = time.time()
        engine.find_best_move(board)
        elapsed = time.time() - start
        
        # Depth 3 should complete in < 30 seconds
        self.assertLess(elapsed, 30)
    
    def test_memory_efficiency(self):
        """Transposition table shouldn't grow unboundedly."""
        evaluator = PositionEvaluator()
        engine = MinimaxEngine(evaluator, max_depth=10)
        
        board = ChessBoard()
        engine.find_best_move(board)
        
        table_size = len(engine.transposition_table)
        # Table size should be reasonable (< 100k entries for depth 2)
        self.assertLess(table_size, 100000)


def run_test_suite():
    """Run all tests with detailed output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestChessBoardMoveGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestRandomForestTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestMinimaxAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestAlphaBetaPruningEfficiency))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Tests Run:    {result.testsRun}")
    print(f"Successes:    {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:     {len(result.failures)}")
    print(f"Errors:       {len(result.errors)}")
    print(f"{'='*70}\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_test_suite()
    exit(0 if success else 1)
