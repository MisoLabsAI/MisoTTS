import unittest

import torch

from generator import _match_num_samples, _stack_audio_frames


class StreamingGenerationHelpersTest(unittest.TestCase):
    def test_stack_audio_frames_preserves_time_order(self) -> None:
        frames = [
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4, 5, 6]]),
        ]

        codes = _stack_audio_frames(frames)

        expected = torch.tensor([[[1, 4], [2, 5], [3, 6]]])
        self.assertTrue(torch.equal(codes, expected))

    def test_match_num_samples_trims_or_pads(self) -> None:
        audio = torch.tensor([1.0, 2.0, 3.0])

        trimmed = _match_num_samples(audio, 2)
        padded = _match_num_samples(audio, 5)

        self.assertTrue(torch.equal(trimmed, torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(padded, torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
