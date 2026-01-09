"""Tests for the command_console module."""

import pytest

from mesa.visualization.command_console import (
    CaptureOutput,
    ConsoleEntry,
    ConsoleManager,
    InteractiveConsole,
    format_command_html,
    format_output_html,
)


class TestConsoleEntry:
    """Tests for the ConsoleEntry dataclass."""

    def test_console_entry_creation_defaults(self):
        """Test ConsoleEntry with default values."""
        entry = ConsoleEntry(command="print('hello')")
        assert entry.command == "print('hello')"
        assert entry.output == ""
        assert entry.is_error is False
        assert entry.is_continuation is False

    def test_console_entry_creation_with_output(self):
        """Test ConsoleEntry with output specified."""
        entry = ConsoleEntry(command="1 + 1", output="2")
        assert entry.command == "1 + 1"
        assert entry.output == "2"

    def test_console_entry_creation_with_error(self):
        """Test ConsoleEntry marked as error."""
        entry = ConsoleEntry(
            command="invalid_var",
            output="NameError: name 'invalid_var' is not defined",
            is_error=True,
        )
        assert entry.is_error is True

    def test_console_entry_creation_continuation(self):
        """Test ConsoleEntry marked as continuation."""
        entry = ConsoleEntry(
            command="    print(x)",
            is_continuation=True,
        )
        assert entry.is_continuation is True

    def test_console_entry_repr(self):
        """Test ConsoleEntry __repr__ method."""
        entry = ConsoleEntry(command="test", output="result", is_error=False, is_continuation=True)
        repr_str = repr(entry)
        assert "ConsoleEntry" in repr_str
        assert "test" in repr_str
        assert "result" in repr_str


class TestCaptureOutput:
    """Tests for the CaptureOutput context manager."""

    def test_capture_stdout(self):
        """Test capturing standard output."""
        with CaptureOutput() as capture:
            print("Hello, World!")
        output, error = capture.get_output()
        assert "Hello, World!" in output
        assert error == ""

    def test_capture_stderr(self):
        """Test capturing standard error."""
        import sys

        with CaptureOutput() as capture:
            print("Error message", file=sys.stderr)
        output, error = capture.get_output()
        assert output == ""
        assert "Error message" in error

    def test_capture_both_stdout_stderr(self):
        """Test capturing both stdout and stderr."""
        import sys

        with CaptureOutput() as capture:
            print("Normal output")
            print("Error output", file=sys.stderr)
        output, error = capture.get_output()
        assert "Normal output" in output
        assert "Error output" in error

    def test_capture_output_clears_after_get(self):
        """Test that get_output clears the buffers."""
        with CaptureOutput() as capture:
            print("First message")
        output1, _ = capture.get_output()
        output2, _ = capture.get_output()
        assert "First message" in output1
        assert output2 == ""

    def test_capture_restores_streams(self):
        """Test that original stdout/stderr are restored after context."""
        import sys

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with CaptureOutput():
            pass

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr


class TestInteractiveConsole:
    """Tests for the InteractiveConsole class."""

    def test_console_initialization_empty_locals(self):
        """Test console initializes with empty locals dict."""
        console = InteractiveConsole()
        assert isinstance(console.capturer, CaptureOutput)

    def test_console_initialization_with_locals(self):
        """Test console initializes with provided locals dict."""
        locals_dict = {"x": 10, "y": 20}
        console = InteractiveConsole(locals_dict)
        more, (output, error) = console.push("x + y")
        assert more is False
        assert "30" in output or output == ""  # Result may be returned

    def test_console_push_simple_expression(self):
        """Test pushing a simple expression."""
        console = InteractiveConsole()
        more, (output, error) = console.push("2 + 2")
        assert more is False

    def test_console_push_multiline_starts(self):
        """Test that multiline statements return more=True."""
        console = InteractiveConsole()
        more, _ = console.push("def foo():")
        assert more is True

    def test_console_push_multiline_completion(self):
        """Test completing a multiline statement."""
        console = InteractiveConsole()
        console.push("def foo():")
        console.push("    return 42")
        more, _ = console.push("")  # Empty line completes
        assert more is False


class TestConsoleManager:
    """Tests for the ConsoleManager class."""

    def test_manager_initialization_no_model(self):
        """Test ConsoleManager initializes without a model."""
        manager = ConsoleManager()
        assert manager.history == []
        assert manager.buffer == []
        assert manager.history_index == -1
        assert manager.current_input == ""

    def test_manager_initialization_with_model(self):
        """Test ConsoleManager initializes with a model."""
        mock_model = type("Model", (), {"step": lambda self: None})()
        manager = ConsoleManager(model=mock_model)
        assert "model" in manager.locals_dict
        assert manager.locals_dict["model"] is mock_model

    def test_manager_initialization_with_additional_imports(self):
        """Test ConsoleManager initializes with additional imports."""
        imports = {"numpy": "np_module", "custom_var": 42}
        manager = ConsoleManager(additional_imports=imports)
        assert manager.locals_dict["numpy"] == "np_module"
        assert manager.locals_dict["custom_var"] == 42

    def test_execute_simple_print(self):
        """Test executing a simple print statement."""
        manager = ConsoleManager()
        callback_calls = []

        def set_input_callback(text):
            callback_calls.append(text)

        manager.execute_code("print('hello')", set_input_callback)
        assert len(manager.history) == 1
        assert manager.history[0].command == "print('hello')"
        assert "hello" in manager.history[0].output
        assert callback_calls == [""]

    def test_execute_expression(self):
        """Test executing an expression."""
        manager = ConsoleManager()

        def set_input_callback(text):
            pass

        manager.execute_code("x = 5", set_input_callback)
        assert len(manager.history) == 1
        assert manager.history[0].command == "x = 5"

    def test_execute_error_command(self):
        """Test executing a command that causes an error."""
        manager = ConsoleManager()

        def set_input_callback(text):
            pass

        manager.execute_code("undefined_variable", set_input_callback)
        assert len(manager.history) == 1
        assert manager.history[0].is_error is True
        assert "NameError" in manager.history[0].output

    def test_history_command(self):
        """Test the 'history' special command."""
        manager = ConsoleManager()

        def set_input_callback(text):
            pass

        manager.execute_code("x = 1", set_input_callback)
        manager.execute_code("y = 2", set_input_callback)
        manager.execute_code("history", set_input_callback)

        assert len(manager.history) == 3
        assert manager.history[2].command == "[history]"
        assert "x = 1" in manager.history[2].output
        assert "y = 2" in manager.history[2].output

    def test_cls_command(self):
        """Test the 'cls' (clear) special command."""
        manager = ConsoleManager()

        def set_input_callback(text):
            pass

        manager.execute_code("x = 1", set_input_callback)
        manager.execute_code("y = 2", set_input_callback)
        assert len(manager.history) == 2

        manager.execute_code("cls", set_input_callback)
        assert len(manager.history) == 0

    def test_tips_command(self):
        """Test the 'tips' special command."""
        manager = ConsoleManager()

        def set_input_callback(text):
            pass

        manager.execute_code("tips", set_input_callback)
        assert len(manager.history) == 1
        assert manager.history[0].command == "[tips]"
        assert "Available Console Commands" in manager.history[0].output

    def test_empty_line_without_buffer(self):
        """Test empty line without any buffer."""
        manager = ConsoleManager()

        def set_input_callback(text):
            pass

        manager.execute_code("", set_input_callback)
        assert len(manager.history) == 1
        assert manager.history[0].command == ""

    def test_clear_console(self):
        """Test clear_console method."""
        manager = ConsoleManager()

        def set_input_callback(text):
            pass

        manager.execute_code("x = 1", set_input_callback)
        manager.buffer = ["some", "buffer"]
        manager.history_index = 5
        manager.current_input = "test input"

        manager.clear_console()

        assert manager.history == []
        assert manager.buffer == []
        assert manager.history_index == -1
        assert manager.current_input == ""

    def test_get_entries(self):
        """Test get_entries returns the history."""
        manager = ConsoleManager()

        def set_input_callback(text):
            pass

        manager.execute_code("x = 1", set_input_callback)
        manager.execute_code("y = 2", set_input_callback)

        entries = manager.get_entries()
        assert len(entries) == 2
        assert entries is manager.history

    def test_prev_command_empty_history(self):
        """Test prev_command with empty history does nothing."""
        manager = ConsoleManager()
        callback_calls = []

        def set_input_callback(text):
            callback_calls.append(text)

        manager.prev_command("current", set_input_callback)
        assert callback_calls == []

    def test_prev_command_navigation(self):
        """Test navigating to previous commands."""
        manager = ConsoleManager()

        def set_input_callback(text):
            pass

        manager.execute_code("cmd1", set_input_callback)
        manager.execute_code("cmd2", set_input_callback)

        callback_results = []

        def capture_callback(text):
            callback_results.append(text)

        manager.prev_command("current_input", capture_callback)
        assert callback_results[-1] == "cmd2"

        manager.prev_command("", capture_callback)
        assert callback_results[-1] == "cmd1"

    def test_next_command_not_in_history_mode(self):
        """Test next_command when not in history navigation mode."""
        manager = ConsoleManager()
        callback_calls = []

        def set_input_callback(text):
            callback_calls.append(text)

        manager.next_command(set_input_callback)
        assert callback_calls == []

    def test_next_command_navigation(self):
        """Test navigating forward through commands."""
        manager = ConsoleManager()

        def set_input_callback(text):
            pass

        manager.execute_code("cmd1", set_input_callback)
        manager.execute_code("cmd2", set_input_callback)

        callback_results = []

        def capture_callback(text):
            callback_results.append(text)

        # Go back
        manager.prev_command("current", capture_callback)
        manager.prev_command("", capture_callback)

        # Go forward
        manager.next_command(capture_callback)
        assert callback_results[-1] == "cmd2"


class TestFormatFunctions:
    """Tests for HTML formatting functions."""

    def test_format_command_html_normal(self):
        """Test formatting a normal command."""
        entry = ConsoleEntry(command="print('hello')")
        html = format_command_html(entry)
        assert ">>>" in html
        assert "print" in html or "hello" in html

    def test_format_command_html_continuation(self):
        """Test formatting a continuation line."""
        entry = ConsoleEntry(command="    return x", is_continuation=True)
        html = format_command_html(entry)
        assert "..:" in html

    def test_format_command_html_multiline(self):
        """Test formatting a multiline command."""
        entry = ConsoleEntry(command="def foo():\n    return 42")
        html = format_command_html(entry)
        assert ">>>" in html
        assert "def foo" in html or "return" in html

    def test_format_output_html_empty(self):
        """Test formatting with empty output."""
        entry = ConsoleEntry(command="x = 1", output="")
        html = format_output_html(entry)
        assert "</div>" in html

    def test_format_output_html_with_output(self):
        """Test formatting with output."""
        entry = ConsoleEntry(command="print('test')", output="test")
        html = format_output_html(entry)
        assert "test" in html

    def test_format_output_html_error(self):
        """Test formatting error output."""
        entry = ConsoleEntry(
            command="undefined",
            output="NameError: name 'undefined' is not defined",
            is_error=True,
        )
        html = format_output_html(entry)
        assert "#ff3860" in html  # Error color
        assert "NameError" in html

    def test_format_output_html_escapes_special_chars(self):
        """Test that special HTML characters are escaped."""
        entry = ConsoleEntry(command="test", output="<script>alert('xss')</script>")
        html = format_output_html(entry)
        assert "&lt;script&gt;" in html
        assert "<script>" not in html
