import unittest
from unittest.mock import patch, MagicMock
import numpy as np

class TestModels(unittest.TestCase):
    @patch('model_utils.AutoProcessor')
    @patch('model_utils.AutoModelForCausalLM')
    def test_generate_captions_call(self, mock_model, mock_processor):
        """
        Test that the generate_captions function is called with the correct arguments.
        """
        # Create a dummy video (list of numpy arrays)
        dummy_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]

        # Mock the return value
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        # Call the function
        from model_utils import generate_captions
        generate_captions(dummy_frames, "cpu")

        # Check that the function was called
        mock_processor.from_pretrained.assert_called_with("microsoft/git-base-vatex")
        mock_model.from_pretrained.assert_called_with("microsoft/git-base-vatex")


    @patch('model_utils.pipeline')
    @patch('model_utils.AutoTokenizer')
    @patch('model_utils.AutoModelForSeq2SeqLM')
    def test_generate_long_summary_call(self, mock_model, mock_tokenizer, mock_pipeline):
        """
        Test that the generate_long_summary function is called with the correct arguments.
        """
        # Create dummy data
        dummy_events = ["drive", "turn_left", "drive"]
        dummy_landmarks = ["store", "none", "gas station"]
        dummy_captions = ["a car is driving down a street", "a car is turning left", "a car is passing a gas station"]

        # Mock the return value
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        mock_pipeline.return_value = MagicMock()

        # Call the function
        from model_utils import generate_long_summary
        generate_long_summary(dummy_events, dummy_landmarks, dummy_captions)

        # Check that the function was called
        mock_tokenizer.from_pretrained.assert_called_with("google/pegasus-cnn_dailymail", cache_dir="models/google_pegasus-cnn_dailymail")
        mock_model.from_pretrained.assert_called_with("google/pegasus-cnn_dailymail", cache_dir="models/google_pegasus-cnn_dailymail")

if __name__ == "__main__":
    unittest.main()
