import json
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import review_transcript


class FakeResponse:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def raise_for_status(self):
        pass

    def json(self):
        return self.events[0]

    def iter_lines(self):
        events = self.events
        if not any("done" in event or "done_reason" in event for event in events):
            events = [*events, {"done": True, "done_reason": "stop"}]
        return iter(json.dumps(event).encode("utf-8") for event in events)


class ReplacementTests(unittest.TestCase):
    def test_global_simultaneous_replacements_and_overlapping_terms(self):
        response = json.dumps([
            {"old": "ab", "new": "x"},
            {"old": "a", "new": "ab"},
            {"old": "x", "new": "y"},
        ])
        self.assertEqual(review_transcript.apply_replacements("ab a x ab", response), "x ab y x")

    def test_empty_list_leaves_text_unchanged(self):
        self.assertEqual(review_transcript.apply_replacements("unchanged", "[]"), "unchanged")

    def test_rejects_invalid_or_unusable_replacements(self):
        cases = [
            "not JSON",
            "{}",
            '[{"old":"a"}]',
            '[{"old":"","new":"b"}]',
            '[{"old":"a","new":null}]',
            '[{"old":"missing","new":"b"}]',
            '[{"old":"a","new":"b"},{"old":"a","new":"c"}]',
            '[{"old":"ab","new":"b"},{"old":"a","new":"c"}]',
        ]
        for response in cases:
            with self.subTest(response=response), self.assertRaises(ValueError):
                review_transcript.apply_replacements("ab", response)


class ClearTests(unittest.TestCase):
    def test_removes_original_lines_in_any_order_and_preserves_line_endings(self):
        text = "first\r\n\r\nthird\nlast"
        self.assertEqual(
            review_transcript.number_lines(text),
            "1\tfirst\r\n2\t\r\n3\tthird\n4\tlast",
        )
        self.assertEqual(review_transcript.clear_lines(text, "[3, 1]"), "\r\nlast")
        self.assertEqual(review_transcript.clear_lines(text, "[1, 3]"), "\r\nlast")
        self.assertEqual(review_transcript.clear_lines(text, "[1, 2, 3, 4]"), "")
        self.assertEqual(review_transcript.clear_lines(text, "[]"), text)
        self.assertEqual(review_transcript.clear_lines("", "[]"), "")

    def test_rejects_invalid_line_lists(self):
        for response in ("not JSON", "{}", "[0]", "[-1]", "[3]", "[1, 1]",
                         "[true]", "[1.0]", '["1"]', "[null]"):
            with self.subTest(response=response), self.assertRaises(ValueError):
                review_transcript.clear_lines("first\nsecond", response)


class GenerationTests(unittest.TestCase):
    def test_length_limit_rejects_empty_or_partial_response(self):
        for events, stream in (
            ([{"thinking": "analyzing"}, {"done_reason": "length"}], True),
            ([{"response": "[1", "done_reason": "length"}], True),
            ([{"response": "partial", "done_reason": "length"}], False),
        ):
            with self.subTest(events=events, stream=stream):
                with patch.object(
                    review_transcript.requests, "post", return_value=FakeResponse(events)
                ):
                    with self.assertRaisesRegex(ValueError, "done_reason: length"):
                        review_transcript.generate_review({}, stream)

    def test_thinking_without_response_reports_completion_reason(self):
        events = [{"thinking": "analyzing"}, {"done_reason": "stop"}]
        with patch.object(
            review_transcript.requests, "post", return_value=FakeResponse(events)
        ):
            with self.assertRaisesRegex(ValueError, "no response text.*done_reason: stop"):
                review_transcript.generate_review({}, True)

    def test_stream_error_is_reported(self):
        with patch.object(
            review_transcript.requests, "post",
            return_value=FakeResponse([{"error": "model failed"}]),
        ):
            with self.assertRaisesRegex(ValueError, "Ollama generation error: model failed"):
                review_transcript.generate_review({}, True)

    def test_clear_stream_stops_at_first_out_of_range_number(self):
        class TrackingResponse(FakeResponse):
            def __init__(self):
                super().__init__([])
                self.consumed = 0
                self.closed = False

            def __exit__(self, exc_type, exc_value, traceback):
                self.closed = True

            def iter_lines(self):
                for chunk in ("[1, 2,", "3", ",4,5]"):
                    self.consumed += 1
                    yield json.dumps({"response": chunk}).encode("utf-8")

        response = TrackingResponse()
        with patch.object(review_transcript.requests, "post", return_value=response):
            with self.assertRaisesRegex(review_transcript.ClearResponseError, "out of range") as error:
                review_transcript.generate_review({}, True, clear_line_count=2)
        self.assertEqual(error.exception.partial_response, "[1, 2,3")
        self.assertEqual(response.consumed, 2)
        self.assertTrue(response.closed)

    def test_clear_stream_validates_chunk_boundaries_and_duplicates(self):
        for chunks, message in (
            (["[", "1", ",", "1", ",", "2]"], "repeated"),
            (["[", "1", ",", "0", "]"], "Invalid"),
            (["[", "t", "rue]"], "Invalid"),
            (["[]", " " * 1025], "excessive whitespace"),
        ):
            with self.subTest(chunks=chunks):
                events = [{"response": chunk} for chunk in chunks]
                with patch.object(
                    review_transcript.requests, "post", return_value=FakeResponse(events)
                ):
                    with self.assertRaisesRegex(review_transcript.ClearResponseError, message):
                        review_transcript.generate_review({}, True, clear_line_count=2)

    def test_clear_stream_checks_actual_line_limit_across_chunks(self):
        validator = review_transcript.ClearResponseValidator(4117)
        validator.feed("[411")
        validator.feed("7, 411")
        with self.assertRaisesRegex(ValueError, "4118"):
            validator.feed("8")

    def test_clear_forces_streaming_without_changing_nonstream_display(self):
        class QuietResponse(FakeResponse):
            def iter_lines(self):
                for event in ({"response": "["}, {"response": "1"}, {"response": "]"},
                              {"done": True}):
                    self_test.assertEqual(stdout.getvalue(), "")
                    yield json.dumps(event).encode("utf-8")

        self_test = self
        with (
            patch.object(review_transcript.requests, "post", return_value=QuietResponse([])) as post,
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            self.assertEqual(review_transcript.generate_review(
                {"stream": False}, False, clear_line_count=1
            ), "[1]")
        self.assertTrue(post.call_args.kwargs["stream"])
        self.assertTrue(post.call_args.kwargs["json"]["stream"])
        self.assertIn("[1]", stdout.getvalue())

    def test_clear_rejects_stream_without_completion_event(self):
        events = [{"response": "[]", "done": False}]
        with patch.object(
            review_transcript.requests, "post", return_value=FakeResponse(events)
        ):
            with self.assertRaisesRegex(review_transcript.ClearResponseError, "completion event"):
                review_transcript.generate_review({}, True, clear_line_count=1)


class ReviewPipelineTests(unittest.TestCase):
    def setUp(self):
        self.original_directory = Path.cwd()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        os.chdir(self.root)
        self.addCleanup(self.temporary_directory.cleanup)
        self.addCleanup(os.chdir, self.original_directory)
        self.prompts = self.root / "prompts"
        self.prompts.mkdir()
        self.default_prompt = self.root / "default.md"
        self.default_prompt.write_text("default: {transcript}", encoding="utf-8")
        (self.root / "input.txt").write_text("original", encoding="utf-8")
        self.prompt_patch = patch.object(review_transcript, "PROMPTS_DIRECTORY", self.prompts)
        self.default_patch = patch.object(
            review_transcript, "DEFAULT_PROMPT_FILE", self.default_prompt
        )
        self.prompt_patch.start()
        self.default_patch.start()
        self.addCleanup(self.prompt_patch.stop)
        self.addCleanup(self.default_patch.stop)

    def run_pipeline(self, responses, stream=False, think=True):
        calls = []

        def post(url, json, stream):
            calls.append((url, json.copy(), stream))
            return FakeResponse(responses[len(calls) - 1])

        with (
            patch.object(review_transcript, "load_model_config", return_value={
                "model": "test", "stream": stream, "think": think,
            }),
            patch.object(review_transcript.requests, "post", side_effect=post),
            patch("sys.argv", ["review_transcript.py", "input.txt"]),
        ):
            review_transcript.main()
        return calls

    def test_pipeline_orders_prompts_chains_responses_and_overwrites_intermediates(self):
        for name in ("03-last.md", "01-first.md", "02-middle.md"):
            (self.prompts / name).write_text(f"{name}: {{transcript}}", encoding="utf-8")
        (self.prompts / "ignored-directory").mkdir()
        temp = self.root / "temp"
        temp.mkdir()
        (temp / "01_01-first.md.txt").write_text("old", encoding="utf-8")

        calls = self.run_pipeline([
            [{"thinking": "private", "response": "first"}],
            [{"response": "second"}],
            [{"response": "final"}],
        ])

        self.assertEqual(
            [call[1]["prompt"] for call in calls],
            ["01-first.md: original", "02-middle.md: first", "03-last.md: second"],
        )
        self.assertEqual((temp / "01_01-first.md.txt").read_text(encoding="utf-8"), "first")
        self.assertEqual((temp / "02_02-middle.md.txt").read_text(encoding="utf-8"), "second")
        self.assertFalse((temp / "03_03-last.md.txt").exists())
        self.assertEqual((self.root / "output" / "input_summary.txt").read_text(encoding="utf-8"), "final")

    def test_fallback_and_streaming(self):
        calls = self.run_pipeline(
            [[{"thinking": "private"}, {"response": "one"}, {"response": " two"}]],
            stream=True,
        )
        self.assertEqual(calls[0][1]["prompt"], "default: original")
        self.assertTrue(calls[0][2])
        self.assertEqual((self.root / "output" / "input_summary.txt").read_text(encoding="utf-8"), "one two")
        self.assertFalse((self.root / "temp").exists())

    def test_next_step_starts_on_a_new_line_after_response(self):
        for name in ("01.md", "02.md"):
            (self.prompts / name).write_text("{transcript}", encoding="utf-8")
        for stream in (False, True):
            with self.subTest(stream=stream), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.run_pipeline(
                    [[{"response": "first"}], [{"response": "second"}]], stream=stream
                )
                self.assertIn("first\nSending step 2/2", stdout.getvalue())
                self.assertEqual(
                    (self.root / "output" / "input_summary.txt").read_text(encoding="utf-8"),
                    "second",
                )

    def test_invalid_later_prompt_fails_before_first_request(self):
        (self.prompts / "01.md").write_text("{transcript}", encoding="utf-8")
        (self.prompts / "02.md").write_text("missing placeholder", encoding="utf-8")
        with (
            patch.object(review_transcript.requests, "post") as post,
            patch("sys.argv", ["review_transcript.py", "input.txt"]),
        ):
            with self.assertRaisesRegex(ValueError, "02.md"):
                review_transcript.main()
            post.assert_not_called()

    def test_missing_prompts_directory_uses_default(self):
        self.prompts.rmdir()
        self.assertEqual(review_transcript.select_prompt_files(), [self.default_prompt])

    def test_header_tags_are_removed_but_body_text_is_preserved(self):
        (self.prompts / "01.md").write_text(
            "\n[clear] [no-think]\n\nentrada: {transcript}\n[clear]\n",
            encoding="utf-8",
        )
        calls = self.run_pipeline([[{"response": "[]"}]])
        self.assertEqual(
            calls[0][1]["prompt"],
            "Total de linhas da entrada: 1.\n\nentrada: 1\toriginal\n[clear]\n",
        )
        self.assertEqual(calls[0][1]["format"]["items"]["maximum"], 1)
        self.assertEqual(calls[0][1]["format"]["maxItems"], 1)
        self.assertIs(calls[0][1]["think"], False)

    def test_default_prompt_supports_header_tags(self):
        self.default_prompt.write_text("[no-think]\ndefault: {transcript}", encoding="utf-8")
        calls = self.run_pipeline([[{"response": "final"}]])
        self.assertEqual(calls[0][1]["prompt"], "default: original")
        self.assertIs(calls[0][1]["think"], False)

    def test_invalid_headers_and_legacy_filenames_fail_before_generation(self):
        for name, content, message in (
            ("01.md", "[unknown]\n{transcript}", "Unknown prompt tag"),
            ("01.md", "[batch]\n{transcript}", "Unknown prompt tag"),
            ("01.md", "[clear][batch=500,overlap=20]\n{transcript}", "Unknown prompt tag"),
            ("01.md", "[clear]\n[clear]\n{transcript}", "Duplicate prompt tag"),
            ("01.md", "[clear] {transcript}", "whole lines"),
            ("01.md", "[unknown] [clear] {transcript}", "whole lines"),
            ("01[clear].md", "{transcript}", "Move tags from the prompt filename"),
        ):
            with self.subTest(name=name, content=content):
                prompt = self.prompts / name
                prompt.write_text(content, encoding="utf-8")
                with (
                    patch.object(review_transcript.requests, "post") as post,
                    patch("sys.argv", ["review_transcript.py", "input.txt"]),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        review_transcript.main()
                    post.assert_not_called()
                prompt.unlink()

    def test_replace_stage_passes_transformed_text_and_saves_response(self):
        first = "01-identificar.md"
        (self.prompts / first).write_text(
            "[no-think][replace]\nidentificar: {transcript}", encoding="utf-8"
        )
        (self.prompts / "02.md").write_text("resumir: {transcript}", encoding="utf-8")
        original = "[SPEAKER_00]: A\n[SPEAKER_00]: B\n[SPEAKER_01]: C\n"
        (self.root / "input.txt").write_text(original, encoding="utf-8")
        replacements = '[{"old":"[SPEAKER_00]:","new":"[Ton]:"}]'
        calls = self.run_pipeline([
            [{"response": replacements}],
            [{"response": "final"}],
        ], stream=True)
        changed = "[Ton]: A\n[Ton]: B\n[SPEAKER_01]: C\n"
        self.assertEqual(calls[0][1]["prompt"], "identificar: " + original)
        self.assertEqual(calls[0][1]["format"], review_transcript.REPLACEMENTS_SCHEMA)
        self.assertIs(calls[0][1]["think"], False)
        self.assertEqual(calls[1][1]["prompt"], "resumir: " + changed)
        self.assertNotIn("format", calls[1][1])
        self.assertIs(calls[1][1]["think"], True)
        temp = self.root / "temp"
        self.assertEqual((temp / f"01_{first}.replacements.json").read_text(encoding="utf-8"), replacements)
        self.assertEqual((temp / f"01_{first}.txt").read_text(encoding="utf-8"), changed)
        self.assertEqual((self.root / "output" / "input_summary.txt").read_text(encoding="utf-8"), "final")

    def test_replace_stages_use_previous_stage_output(self):
        for name in ("01.md", "02.md", "03.md", "04.md"):
            header = "[replace]\n" if name in ("02.md", "03.md") else ""
            (self.prompts / name).write_text(header + "{transcript}", encoding="utf-8")
        calls = self.run_pipeline([
            [{"response": "texto processado"}],
            [{"response": '[{"old":"processado","new":"revisado"}]'}],
            [{"response": '[{"old":"revisado","new":"finalizado"}]'}],
            [{"response": "resumo"}],
        ])
        self.assertEqual(
            [call[1]["prompt"] for call in calls],
            ["original", "texto processado", "texto revisado", "texto finalizado"],
        )
        self.assertEqual(
            (self.root / "temp" / "02_02.md.txt").read_text(encoding="utf-8"),
            "texto revisado",
        )
        self.assertEqual(
            (self.root / "temp" / "03_03.md.txt").read_text(encoding="utf-8"),
            "texto finalizado",
        )
        self.assertEqual((self.root / "input.txt").read_text(encoding="utf-8"), "original")

    def test_clear_stage_filters_previous_output_and_passes_unnumbered_text(self):
        for name in ("01.md", "02.md", "03.md"):
            header = "[no-think]\n[clear]\n" if name == "02.md" else ""
            (self.prompts / name).write_text(header + "input: {transcript}", encoding="utf-8")
        calls = self.run_pipeline([
            [{"response": "alpha\nbeta\ngamma\ndelta"}],
            [{"response": "[4, 2]"}],
            [{"response": "resumo"}],
        ], stream=True)
        self.assertEqual(calls[0][1]["prompt"], "input: original")
        self.assertEqual(
            calls[1][1]["prompt"],
            "Total de linhas da entrada: 4.\n\ninput: 1\talpha\n2\tbeta\n3\tgamma\n4\tdelta",
        )
        self.assertEqual(calls[1][1]["format"]["items"]["maximum"], 4)
        self.assertEqual(calls[1][1]["format"]["maxItems"], 4)
        self.assertIs(calls[1][1]["think"], False)
        self.assertEqual(calls[2][1]["prompt"], "input: alpha\ngamma\n")
        self.assertNotIn("format", calls[2][1])
        temp = self.root / "temp"
        name = "02_02.md"
        self.assertEqual((temp / f"{name}.clear.json").read_text(encoding="utf-8"), "[4, 2]")
        self.assertEqual((temp / f"{name}.txt").read_text(encoding="utf-8"), "alpha\ngamma\n")
        self.assertEqual((self.root / "input.txt").read_text(encoding="utf-8"), "original")

    def test_clear_sends_large_input_in_one_request(self):
        (self.prompts / "01.md").write_text("[clear]\n{transcript}", encoding="utf-8")
        lines = [f"line {number}\n" for number in range(1, 542)]
        (self.root / "input.txt").write_text("".join(lines), encoding="utf-8")
        calls = self.run_pipeline([[{"response": "[500, 501]"}]])
        self.assertEqual(len(calls), 1)
        self.assertIn("Total de linhas da entrada: 541.", calls[0][1]["prompt"])
        self.assertIn("500\tline 500\n501\tline 501\n", calls[0][1]["prompt"])
        self.assertIn("541\tline 541\n", calls[0][1]["prompt"])
        self.assertEqual(calls[0][1]["format"]["items"]["maximum"], 541)
        self.assertEqual(
            (self.root / "output" / "input_summary.txt").read_text(encoding="utf-8"),
            "".join(lines[:499] + lines[501:]),
        )

    def test_invalid_clear_response_stops_pipeline_and_preserves_response(self):
        (self.prompts / "01.md").write_text("[clear]\n{transcript}", encoding="utf-8")
        (self.prompts / "02.md").write_text("{transcript}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.run_pipeline([[{"response": "[2]"}]])
        self.assertEqual(
            (self.root / "temp" / "01_01.md.clear.json").read_text(encoding="utf-8"),
            "[2]",
        )
        self.assertFalse((self.root / "temp" / "01_01.md.txt").exists())
        self.assertFalse((self.root / "output" / "input_summary.txt").exists())

    def test_consecutive_clear_stages_only_number_their_own_input(self):
        for name in ("01.md", "02.md", "03.md", "04.md"):
            header = "[clear]\n" if name in ("02.md", "03.md") else ""
            (self.prompts / name).write_text(header + "{transcript}", encoding="utf-8")
        calls = self.run_pipeline([
            [{"response": "123 frutas\n789 legumes\n456 peixe"}],
            [{"response": "[2]"}],
            [{"response": "[2]"}],
            [{"response": "final"}],
        ])
        self.assertEqual(
            calls[1][1]["prompt"],
            "Total de linhas da entrada: 3.\n\n1\t123 frutas\n2\t789 legumes\n3\t456 peixe",
        )
        self.assertEqual(
            calls[2][1]["prompt"], "Total de linhas da entrada: 2.\n\n1\t123 frutas\n2\t456 peixe"
        )
        self.assertEqual(calls[3][1]["prompt"], "123 frutas\n")
        self.assertEqual(
            (self.root / "temp" / "02_02.md.txt").read_text(encoding="utf-8"),
            "123 frutas\n456 peixe",
        )
        self.assertEqual(
            (self.root / "temp" / "03_03.md.txt").read_text(encoding="utf-8"),
            "123 frutas\n",
        )

    def test_final_clear_stage_writes_unnumbered_text(self):
        (self.prompts / "01.md").write_text("[clear]\n{transcript}", encoding="utf-8")
        (self.root / "input.txt").write_text("first\nsecond\n", encoding="utf-8")
        self.run_pipeline([[{"response": "[2]"}]])
        self.assertEqual(
            (self.root / "output" / "input_summary.txt").read_text(encoding="utf-8"),
            "first\n",
        )

    def test_clear_accepts_empty_input_and_empty_list(self):
        (self.prompts / "01.md").write_text("[clear]\n{transcript}", encoding="utf-8")
        (self.root / "input.txt").write_text("", encoding="utf-8")
        calls = self.run_pipeline([[{"response": "[]"}]])
        self.assertEqual(calls[0][1]["prompt"], "Total de linhas da entrada: 0.\n\n")
        self.assertEqual(calls[0][1]["format"]["maxItems"], 0)
        self.assertEqual((self.root / "output" / "input_summary.txt").read_text(encoding="utf-8"), "")

    def test_conflicting_action_tags_fail_before_any_request(self):
        (self.prompts / "01.md").write_text("{transcript}", encoding="utf-8")
        (self.prompts / "02.md").write_text("[replace][clear]\n{transcript}", encoding="utf-8")
        with (
            patch.object(review_transcript.requests, "post") as post,
            patch("sys.argv", ["review_transcript.py", "input.txt"]),
        ):
            with self.assertRaisesRegex(ValueError, "cannot combine"):
                review_transcript.main()
            post.assert_not_called()

    def test_invalid_replace_response_stops_pipeline_and_preserves_response(self):
        first = "01.md"
        (self.prompts / first).write_text("[replace]\n{transcript}", encoding="utf-8")
        (self.prompts / "02.md").write_text("{transcript}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not found"):
            self.run_pipeline([[{"response": '[{"old":"missing","new":"x"}]'}]])
        self.assertTrue((self.root / "temp" / f"01_{first}.replacements.json").is_file())
        self.assertFalse((self.root / "temp" / f"01_{first}.txt").exists())
        self.assertFalse((self.root / "output" / "input_summary.txt").exists())

    def test_final_replace_stage_produces_transformed_output(self):
        (self.prompts / "01.md").write_text("[replace]\n{transcript}", encoding="utf-8")
        self.run_pipeline([[{"response": '[{"old":"original","new":"changed"}]'}]])
        self.assertEqual((self.root / "output" / "input_summary.txt").read_text(encoding="utf-8"), "changed")

    def test_no_think_tag_only_overrides_the_tagged_stage(self):
        for name in ("01-first.md", "02-no-think.md", "03-last.md"):
            header = "[no-think]\n" if name == "01-first.md" else ""
            (self.prompts / name).write_text(header + "{transcript}", encoding="utf-8")
        for think in (True, "high", False, None):
            with self.subTest(think=think):
                calls = self.run_pipeline([
                    [{"response": "first"}],
                    [{"response": "second"}],
                    [{"response": "final"}],
                ], think=think)
                self.assertEqual([call[1]["think"] for call in calls], [False, think, think])
                self.assertEqual([call[1]["prompt"] for call in calls], [
                    "original", "first", "second",
                ])

    def test_empty_intermediate_response_stops_pipeline(self):
        for name in ("01.md", "02.md", "03.md"):
            (self.prompts / name).write_text("{transcript}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no response text"):
            self.run_pipeline([
                [{"response": "first"}],
                [{"thinking": "only thinking"}],
            ])
        self.assertEqual(
            (self.root / "temp" / "01_01.md.txt").read_text(encoding="utf-8"), "first"
        )
        self.assertFalse((self.root / "temp" / "02_02.md.txt").exists())
        self.assertFalse((self.root / "output" / "input_summary.txt").exists())


if __name__ == "__main__":
    unittest.main()
