import argparse
import json
import re
from collections.abc import Iterator
from pathlib import Path

import requests


MODEL_CONFIG_FILE = Path(__file__).resolve().with_name("review_model.json")
DEFAULT_MODEL = "gemma4:12b"
DEFAULT_MODEL_OPTIONS = {
    "num_ctx": 131072,
    "temperature": 0,
}
PROMPTS_DIRECTORY = Path(__file__).resolve().with_name("prompts")
DEFAULT_PROMPT_FILE = Path(__file__).resolve().with_name("default_review_prompt.md")
REPLACEMENTS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"old": {"type": "string"}, "new": {"type": "string"}},
        "required": ["old", "new"],
        "additionalProperties": False,
    },
}
PROMPT_TAGS = {"clear", "replace", "no-think"}


def select_prompt_files() -> list[Path]:
    prompt_files = (
        sorted(path for path in PROMPTS_DIRECTORY.iterdir() if path.is_file())
        if PROMPTS_DIRECTORY.is_dir()
        else []
    )
    return prompt_files or [DEFAULT_PROMPT_FILE]


def load_prompt(prompt_file: Path) -> tuple[set[str], str]:
    if any(f"[{tag}]" in prompt_file.stem for tag in PROMPT_TAGS):
        raise ValueError(f"Move tags from the prompt filename into its header: {prompt_file}")

    text = prompt_file.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tags: set[str] = set()
    header_end = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if tags:
                header_end = index + 1
            continue
        if not re.fullmatch(r"(?:\[[A-Za-z0-9_,=-]+\][ \t]*)+", stripped):
            prefix = re.match(r"(?:\[[A-Za-z0-9_,=-]+\][ \t]*)+", stripped)
            if prefix and any(
                token.split("=", 1)[0].split(",", 1)[0] in PROMPT_TAGS
                for token in re.findall(r"\[([^\]]+)\]", prefix.group())
            ):
                raise ValueError(f"Prompt tags must occupy whole lines: {prompt_file}")
            break
        for token in re.findall(r"\[([A-Za-z0-9_,=-]+)\]", stripped):
            tag = token.split("=", 1)[0].split(",", 1)[0]
            if tag not in PROMPT_TAGS:
                raise ValueError(f"Unknown prompt tag [{token}]: {prompt_file}")
            if tag in tags:
                raise ValueError(f"Duplicate prompt tag [{tag}]: {prompt_file}")
            if token != tag:
                raise ValueError(f"Invalid prompt tag [{token}]: {prompt_file}")
            tags.add(tag)
        header_end = index + 1

    if "clear" in tags and "replace" in tags:
        raise ValueError(f"Prompt cannot combine [replace] and [clear]: {prompt_file}")
    return tags, "".join(lines[header_end:]) if tags else text


def load_model_config() -> dict[str, object]:
    config = json.loads(MODEL_CONFIG_FILE.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Model configuration must be a JSON object: {MODEL_CONFIG_FILE}")

    unknown_keys = config.keys() - {"model", "options", "stream", "think"}
    if unknown_keys:
        raise ValueError(f"Unknown model configuration keys: {', '.join(sorted(unknown_keys))}")

    model = config.get("model", DEFAULT_MODEL)
    options = config.get("options", {})
    stream = config.get("stream", False)
    think = config.get("think", False)
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"'model' must be a non-empty string: {MODEL_CONFIG_FILE}")
    if not isinstance(options, dict):
        raise ValueError(f"'options' must be a JSON object: {MODEL_CONFIG_FILE}")
    if not isinstance(stream, bool):
        raise ValueError(f"'stream' must be a boolean: {MODEL_CONFIG_FILE}")
    if not isinstance(think, (bool, str)) and think is not None:
        raise ValueError(f"'think' must be a boolean, string, or null: {MODEL_CONFIG_FILE}")

    config["model"] = model
    config["options"] = {**DEFAULT_MODEL_OPTIONS, **options}
    config["stream"] = stream
    config["think"] = think
    return config


def iter_generation_events(
    response: requests.Response, stream: bool
) -> Iterator[dict[str, object]]:
    if stream:
        for line in response.iter_lines():
            if line:
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("Ollama returned a non-object stream event.")
                yield event
    else:
        event = response.json()
        if not isinstance(event, dict):
            raise ValueError("Ollama returned a non-object response.")
        yield event


def apply_replacements(text: str, response: str) -> str:
    try:
        replacements = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid replacement list JSON: {error}") from error
    if not isinstance(replacements, list):
        raise ValueError("Replacement list must be a JSON array.")

    mapping: dict[str, str] = {}
    for index, replacement in enumerate(replacements, start=1):
        if not isinstance(replacement, dict) or set(replacement) != {"old", "new"}:
            raise ValueError(f"Replacement {index} must contain only 'old' and 'new'.")
        old = replacement["old"]
        new = replacement["new"]
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError(f"Replacement {index} must have non-empty text 'old' and text 'new'.")
        if old in mapping:
            raise ValueError(f"Replacement {index} repeats an earlier 'old' value.")
        if old not in text:
            raise ValueError(f"Replacement {index} has an 'old' value not found in the input.")
        mapping[old] = new

    if not mapping:
        return text

    pattern = re.compile("|".join(re.escape(old) for old in sorted(mapping, key=len, reverse=True)))
    applied: set[str] = set()

    def replace_match(match: re.Match[str]) -> str:
        old = match.group()
        applied.add(old)
        return mapping[old]

    result = pattern.sub(replace_match, text)
    if len(applied) != len(mapping):
        raise ValueError("A replacement was shadowed by another overlapping 'old' value.")
    return result


def number_lines(text: str) -> str:
    return "".join(
        f"{number}\t{line}"
        for number, line in enumerate(text.splitlines(keepends=True), start=1)
    )


class ClearResponseError(ValueError):
    def __init__(self, message: str, partial_response: str) -> None:
        super().__init__(message)
        self.partial_response = partial_response


class ClearResponseValidator:
    def __init__(self, line_count: int) -> None:
        self.line_count = line_count
        self.state = "start"
        self.number = 0
        self.seen: set[int] = set()
        self.whitespace = 0

    def finish_number(self) -> None:
        if self.number in self.seen:
            raise ValueError(f"Line removal number repeated: {self.number}.")
        self.seen.add(self.number)
        self.state = "after_number"

    def feed(self, text: str) -> None:
        for char in text:
            if char in " \t\r\n":
                self.whitespace += 1
                if self.whitespace > 1024:
                    raise ValueError("Line removal response contains excessive whitespace.")
                if self.state == "number":
                    self.finish_number()
                continue
            self.whitespace = 0
            if self.state == "start" and char == "[":
                self.state = "first"
            elif self.state in ("first", "after_comma") and char in "123456789":
                self.number = int(char)
                self.state = "number"
            elif self.state == "number" and char in "0123456789":
                self.number = self.number * 10 + int(char)
            else:
                if self.state == "number":
                    self.finish_number()
                if self.state == "after_number" and char == ",":
                    self.state = "after_comma"
                elif self.state in ("first", "after_number") and char == "]":
                    self.state = "end"
                else:
                    raise ValueError("Invalid line removal list JSON.")
            if self.state == "number" and self.number > self.line_count:
                raise ValueError(f"Line removal number out of range: {self.number}.")


def parse_clear_numbers(response: str, last_line: int) -> list[int]:
    try:
        line_numbers = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid line removal list JSON: {error}") from error
    if not isinstance(line_numbers, list):
        raise ValueError("Line removal list must be a JSON array.")

    removed: set[int] = set()
    for line_number in line_numbers:
        if not isinstance(line_number, int) or isinstance(line_number, bool):
            raise ValueError("Line removal numbers must be integers.")
        if not 1 <= line_number <= last_line:
            raise ValueError(f"Line removal number out of range: {line_number}.")
        if line_number in removed:
            raise ValueError(f"Line removal number repeated: {line_number}.")
        removed.add(line_number)
    return line_numbers


def clear_lines(text: str, response: str) -> str:
    lines = text.splitlines(keepends=True)
    removed = set(parse_clear_numbers(response, len(lines)))
    return "".join(line for number, line in enumerate(lines, start=1) if number not in removed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review and summarize a transcript using Ollama."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Transcript file to review",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        help="Output file or existing directory (default: output/<transcript-name>_summary.txt)",
    )
    args = parser.parse_args()

    input_file = args.input_file
    if input_file.parent == Path("."):
        transcript_in_folder = Path("transcripts") / input_file
        if transcript_in_folder.is_file():
            input_file = transcript_in_folder
    if not input_file.is_file():
        parser.error(f"transcript file not found: {input_file}")

    default_output_name = f"{input_file.stem}_summary.txt"
    if args.output_file is None:
        output_file = Path("output") / default_output_name
    elif args.output_file.is_dir():
        output_file = args.output_file / default_output_name
    else:
        output_file = args.output_file
    current_text = input_file.read_text(encoding="utf-8")

    prompt_templates = []
    transcript_placeholder = "{transcript}"
    for prompt_file in select_prompt_files():
        tags, prompt_template = load_prompt(prompt_file)
        if transcript_placeholder not in prompt_template:
            raise ValueError(
                f"Prompt template must include {transcript_placeholder!r}: {prompt_file}"
            )
        prompt_templates.append((prompt_file, tags, prompt_template))

    model_config = load_model_config()
    stream = model_config["stream"]
    if not isinstance(stream, bool):
        raise ValueError("'stream' must be a boolean.")

    for index, (prompt_file, tags, prompt_template) in enumerate(prompt_templates, start=1):
        is_clear = "clear" in tags
        is_replace = "replace" in tags
        stage_config = dict(model_config)
        if "no-think" in tags:
            stage_config["think"] = False
        if index > 1:
            print()
        print(
            f"Sending step {index}/{len(prompt_templates)} "
            f"({prompt_file.name}) to Ollama ({model_config['model']})..."
        )
        line_count = len(current_text.splitlines()) if is_clear else None
        prompt_input = number_lines(current_text) if is_clear else current_text
        prompt = prompt_template.replace(transcript_placeholder, prompt_input)
        if is_replace:
            stage_config["format"] = REPLACEMENTS_SCHEMA
        elif line_count is not None:
            stage_config["format"] = {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": max(1, line_count)},
                "maxItems": line_count,
                "uniqueItems": True,
            }
            prompt = f"Total de linhas da entrada: {line_count}.\n\n{prompt}"
        stage_config["prompt"] = prompt
        try:
            response_text = generate_review(stage_config, stream, clear_line_count=line_count)
        except ClearResponseError as error:
            temp_file = Path("temp") / f"{index:02d}_{prompt_file.name}.clear.json"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(error.partial_response, encoding="utf-8")
            raise
        if is_replace:
            temp_file = Path("temp") / f"{index:02d}_{prompt_file.name}.replacements.json"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(response_text, encoding="utf-8")
            current_text = apply_replacements(current_text, response_text)
        elif is_clear:
            temp_file = Path("temp") / f"{index:02d}_{prompt_file.name}.clear.json"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(response_text, encoding="utf-8")
            current_text = clear_lines(current_text, response_text)
        else:
            current_text = response_text
        if index < len(prompt_templates):
            temp_file = Path("temp") / f"{index:02d}_{prompt_file.name}.txt"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(current_text, encoding="utf-8")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(current_text, encoding="utf-8")

    print(f"\nReview complete! Output file: {output_file}")


def generate_review(
    model_config: dict[str, object],
    stream: bool,
    clear_line_count: int | None = None,
) -> str:
    request_stream = stream or clear_line_count is not None
    request_config = {**model_config, "stream": request_stream}
    clear_validator = (
        ClearResponseValidator(clear_line_count)
        if clear_line_count is not None else None
    )
    with requests.post(
        "http://localhost:11434/api/generate",
        json=request_config,
        stream=request_stream,
    ) as response:
        response.raise_for_status()
        review_parts = []
        thinking_parts = []
        current_section = None
        done_reason = None
        done = False
        for event in iter_generation_events(response, request_stream):
            if "error" in event:
                raise ValueError(f"Ollama generation error: {event['error']}")
            if event.get("done_reason") is not None:
                if not isinstance(event["done_reason"], str):
                    raise ValueError("Ollama returned a non-text completion reason.")
                done_reason = event["done_reason"]
            if event.get("done") is True:
                done = True
            for key, label in (
                ("thinking", "\nThinking:\n"),
                ("response", "\n\nResposta:\n"),
            ):
                text = event.get(key, "")
                if text is None and key == "thinking":
                    continue
                if not isinstance(text, str):
                    raise ValueError(f"Ollama returned non-text content for {key!r}.")
                if text:
                    if key == "response":
                        review_parts.append(text)
                        if clear_validator is not None:
                            try:
                                clear_validator.feed(text)
                            except ValueError as error:
                                raise ClearResponseError(
                                    str(error), "".join(review_parts)
                                ) from error
                    elif not stream:
                        thinking_parts.append(text)
                    if stream:
                        if current_section != key:
                            print(label, end="", flush=True)
                            current_section = key
                        print(text, end="", flush=True)
        review = "".join(review_parts)
        if done_reason == "length":
            raise ValueError(
                "Ollama stopped at a generation or context limit (done_reason: length); "
                "the response may be incomplete. Reduce the input or check num_ctx and num_predict."
            )
        if clear_validator is not None and not done:
            raise ClearResponseError(
                "Ollama ended the line removal stream without a completion event.", review
            )
        if not review:
            reason = f" (done_reason: {done_reason})" if done_reason else ""
            raise ValueError(
                f"Ollama returned no response text{reason}. "
                "If the model only produced thinking, add [no-think] to the prompt header."
            )
        if not stream:
            if thinking_parts:
                print("\nThinking:\n" + "".join(thinking_parts), end="", flush=True)
            print("\n\nResposta:\n" + review, end="", flush=True)
    return review


if __name__ == "__main__":
    main()
