import io
import os
import sys
import unittest
from unittest import mock

import run_misotts


class RunMisoTTSArgsTest(unittest.TestCase):
    def test_defaults_match_existing_example(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            args = run_misotts.parse_args([])

        self.assertEqual(args.device, "auto")
        self.assertEqual(args.output, "full_conversation.wav")
        self.assertIsNone(args.model)
        self.assertEqual(args.max_audio_length_ms, 10_000)
        self.assertEqual(args.temperature, 0.9)
        self.assertEqual(args.topk, 50)

    def test_model_defaults_to_environment_override(self):
        with mock.patch.dict(os.environ, {"MISO_TTS_8B_MODEL": "local-checkpoint"}):
            args = run_misotts.parse_args([])

        self.assertEqual(args.model, "local-checkpoint")

    def test_accepts_custom_generation_settings(self):
        args = run_misotts.parse_args(
            [
                "--model",
                "MisoLabs/MisoTTS",
                "--device",
                "cpu",
                "--output",
                "sample.wav",
                "--max-audio-length-ms",
                "5000",
                "--temperature",
                "0.7",
                "--topk",
                "25",
            ]
        )

        self.assertEqual(args.model, "MisoLabs/MisoTTS")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.output, "sample.wav")
        self.assertEqual(args.max_audio_length_ms, 5000)
        self.assertEqual(args.temperature, 0.7)
        self.assertEqual(args.topk, 25)

    def test_rejects_non_positive_generation_settings(self):
        with mock.patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                run_misotts.parse_args(["--topk", "0"])

            with self.assertRaises(SystemExit):
                run_misotts.parse_args(["--max-audio-length-ms", "0"])

            with self.assertRaises(SystemExit):
                run_misotts.parse_args(["--temperature", "0"])

    def test_resolve_device_returns_explicit_device(self):
        self.assertEqual(run_misotts.resolve_device("cuda:1"), "cuda:1")


if __name__ == "__main__":
    unittest.main()
