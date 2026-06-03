import io
import os
import sys
import unittest
from unittest import mock

import run_misotts


class RunMisoTTSCliTest(unittest.TestCase):
    def parse_args(self, *args: str):
        with mock.patch.object(sys, "argv", ["run_misotts.py", *args]):
            return run_misotts.parse_args()

    def test_defaults_keep_existing_example_behavior(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            args = self.parse_args()

        self.assertEqual(args.device, "auto")
        self.assertEqual(args.output, "full_conversation.wav")
        self.assertIsNone(args.model)
        self.assertEqual(args.max_audio_length_ms, 10_000)
        self.assertEqual(args.temperature, 0.9)
        self.assertEqual(args.topk, 50)

    def test_model_defaults_to_environment_override(self):
        with mock.patch.dict(os.environ, {"MISO_TTS_8B_MODEL": "local-checkpoint"}):
            args = self.parse_args()

        self.assertEqual(args.model, "local-checkpoint")

    def test_rejects_non_positive_values(self):
        with mock.patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse_args("--topk", "0")

            with self.assertRaises(SystemExit):
                self.parse_args("--max-audio-length-ms", "0")

    def test_explicit_device_does_not_require_auto_detection(self):
        self.assertEqual(run_misotts.resolve_device("cuda:1"), "cuda:1")


if __name__ == "__main__":
    unittest.main()
