"""A command console interface for interactive Python code execution in the browser.

This module provides a set of classes and functions to create an interactive Python console
that can be embedded in a web browser. It supports command history, multi-line code blocks,
and special commands for console management.

Notes:
    - The console supports multi-line code blocks with proper indentation
    - Output is captured and displayed with appropriate formatting
    - Error messages are displayed in red with distinct styling
    - The console maintains a history of commands and their outputs

"""

import code
import io
import sys
from collections.abc import Callable
from dataclasses import dataclass

import solara
from solara.components.input import use_change

from mesa.visualization.utils import force_update


@dataclass
class ConsoleEntry:
    """A class to store command console entries.

    Attributes:
        command (str): The command entered
        output (str): The output of the command
        is_error (bool): Whether the entry represents an error
        is_continuation (bool): Whether the entry is a continuation of previous command

    """

    command: str
    output: str = ""
    is_error: bool = False
    is_continuation: bool = False

    def __repr__(self):
        """Return a string representation of the ConsoleEntry."""
        return f"ConsoleEntry({self.command}, {self.output}, {self.is_error}, {self.is_continuation})"


class CaptureOutput:
    """A context manager for capturing stdout and stderr output.

    This class provides a way to capture output that would normally be printed
    to stdout and stderr during the execution of code within its context.
    """

    def __init__(self):
        """Initialize the CaptureOutput context manager with empty string buffers."""
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self):
        """Set up the context manager by redirecting stdout and stderr.

        Returns:
            self: The context manager instance

        """
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the original stdout and stderr when exiting the context."""
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr

    def get_output(self):
        """Retrieve and clear the captured output.

        Returns:
            tuple: A pair of strings (stdout_output, stderr_output)

        """
        output = self.stdout.getvalue()
        error = self.stderr.getvalue()
        self.stdout.seek(0)
        self.stdout.truncate(0)
        self.stderr.seek(0)
        self.stderr.truncate(0)
        return output, error


class InteractiveConsole(code.InteractiveConsole):
    """A custom interactive Python console with output capturing capabilities.

    This class extends code.InteractiveConsole to provide output capturing functionality
    when executing Python code interactively.

    Args:
        locals_dict (dict, optional): Dictionary of local variables. Defaults to None.

    """

    def __init__(self, locals_dict=None):
        """Initialize the InteractiveConsole with the provided locals dictionary."""
        super().__init__(locals=locals_dict or {})
        self.capturer = CaptureOutput()

    def push(self, line):
        """Push a line to the command interpreter and execute it.

        This method captures the output of the executed command and returns both
        the 'more' flag and the captured output.

        Args:
            line (str): The line of code to be executed.

        Returns:
            tuple: A tuple containing:
                - more (bool): Flag indicating if more input is needed
                - str: The captured output from executing the command

        """
        with self.capturer:
            more = super().push(line)
        return more, self.capturer.get_output()


class ConsoleManager:
    """A console manager for executing Python code interactively.

    This class provides functionality to execute Python code in an interactive console environment,
    maintain command history, and handle multi-line code blocks.

    Attributes:
        locals_dict (dict): Dictionary containing local variables available to the console
        console (InteractiveConsole): Python's interactive console instance
        buffer (list): Buffer for storing multi-line code blocks
        history (list[ConsoleEntry]): List of console entries containing commands and their outputs
    Special Commands:
        1. `history` : Shows the command history
        2. `cls` : Clears the console screen
        3. `tips` : Shows available console commands and usage tips
    Example:
        >>> console = ConsoleManager(model=my_model)
        >>> console.execute_code("print('hello world')", set_input_callback)

    """

    def __init__(self, model=None, additional_imports=None):
        """Initialize the console manager with the provided model and imports."""
        # Create locals dictionary with model and imports
        locals_dict = {}
        if model is not None:
            locals_dict["model"] = model
        if additional_imports:
            locals_dict.update(additional_imports)

        self.locals_dict = locals_dict
        self.console = InteractiveConsole(locals_dict)
        self.buffer = []
        self.history: list[ConsoleEntry] = []
        self.history_index = -1
        self.current_input = ""

    def execute_code(
        self, code_line: str, set_input_text: Callable[[str], None]
    ) -> None:
        """Execute the provided code line and update the console history."""
        # Custom Commands
        # A. History
        if code_line == "history":
            # Get the current history except the custom commands
            cur_his = [
                (
                    f"Command: {entry.command}, \nOutput: {entry.output if entry.output else None}\n"
                )
                for entry in self.history
                if entry.command != "[history]"
                and entry.command != "[tips]"
                and entry.command != ""
            ]

            self.history.append(
                ConsoleEntry(
                    command="[history]",
                    output="\n".join(cur_his) if cur_his else "No history",
                    is_error=False,
                    is_continuation=False,
                )
            )

            set_input_text("")
            return

        # B. Clear
        if code_line == "cls":
            self.clear_console()
            set_input_text("")
            return

        # C. Tips
        if code_line == "tips":
            self.history.append(
                ConsoleEntry(
                    command="[tips]",
                    output="Available Console Commands:\n1. Press Enter to execute a command\n2. Type 'cls' to clear the console screen (it doesn't delete past variables and functions)\n3. Type 'history' to view previous commands\n4. Press Enter on an empty line to complete a multiline block\n5. Use proper indentation for multiline blocks\n6. The console will show '..: ' for continuation lines",
                    is_error=False,
                    is_continuation=False,
                )
            )

            set_input_text("")
            return

        # Handle empty lines
        if not code_line.strip():
            # If we have a buffer, complete the block
            if self.buffer:
                full_code = "\n".join(self.buffer)
                more, (output, error) = self.console.push("")

                # Remove the redundant commands from the history
                for _ in range(len(self.buffer) - 1):
                    self.history.pop()

                # Completing a multi-line block
                if self.history:
                    self.history[-1].command = full_code
                    self.history[-1].output = error if error else output
                    self.history[-1].is_error = bool(error)
                    self.history[-1].is_continuation = False
                self.buffer = []
            else:
                # Empty line with no buffer - just add a blank entry
                self.history.append(ConsoleEntry(command=""))
            set_input_text("")
            return

        # Execute the line
        more, (output, error) = self.console.push(code_line)

        # Force update to display any changes to the model
        force_update()

        # If this is the start of a multi-line block
        if more:
            self.buffer.append(code_line)
            self.history.append(
                ConsoleEntry(
                    command=code_line,
                    output="",  # Don't show partial output for incomplete blocks
                    is_error=False,
                    is_continuation=True,
                )
            )
        else:
            # Single complete command
            if not self.buffer:
                # Normal single-line command
                self.history.append(
                    ConsoleEntry(
                        command=code_line,
                        output=error if error else output,
                        is_error=bool(error),
                        is_continuation=False,
                    )
                )
            else:
                # Remove the redundant commands from the history
                for _ in range(len(self.buffer) - 1):
                    self.history.pop()

                # Completing a multi-line block
                self.buffer.append(code_line)
                full_code = "\n".join(self.buffer)
                if self.history:
                    self.history[-1].command = full_code
                    self.history[-1].output = error if error else output
                    self.history[-1].is_error = bool(error)
                    self.history[-1].is_continuation = False
                self.buffer = []

        set_input_text("")

    def clear_console(self) -> None:
        """Clear the console history and reset the console state."""
        self.history.clear()
        self.buffer.clear()
        self.history_index = -1
        self.current_input = ""
        # Reset the console while maintaining the locals dictionary
        self.console = InteractiveConsole(self.locals_dict)

    def get_entries(self) -> list[ConsoleEntry]:
        """Get the list of console entries."""
        return self.history

    def prev_command(
        self, current_text: str, set_input_text: Callable[[str], None]
    ) -> None:
        """Navigate to previous command in history."""
        if not self.history:
            return

        # Save the current input
        if self.history_index == -1:
            self.current_input = current_text

        # Move up in history
        if self.history_index == -1:
            self.history_index = len(self.history) - 1
        elif self.history_index > 0:
            self.history_index -= 1

        # Set text to the historical command
        if 0 <= self.history_index < len(self.history):
            set_input_text(self.history[self.history_index].command)

    def next_command(self, set_input_text: Callable[[str], None]) -> None:
        """Navigate to next command in history."""
        if self.history_index == -1:
            return  # Not in history navigation mode

        # Move down in history
        self.history_index += 1

        # If we've moved past the end of history, restore the saved input
        if self.history_index >= len(self.history):
            self.history_index = -1
            set_input_text(self.current_input)
        else:
            set_input_text(self.history[self.history_index].command)


def format_command_html(entry):
    """Format the command part of a console entry as HTML."""
    prompt = "..: " if entry.is_continuation else ">>> "
    prompt_color = "#6c757d" if entry.is_continuation else "#2196F3"

    # Handle multi-line commands with proper formatting
    command_lines = entry.command.split("\n")
    formatted_lines = []

    for i, line in enumerate(command_lines):
        line_prompt = prompt if i == 0 else "..: "
        line_prompt_color = prompt_color if i == 0 else "#6c757d"
        formatted_lines.append(
            f'<span style="color: {line_prompt_color};">{line_prompt}</span><span>{line.removeprefix(">>> ").removeprefix("..: ")}</span>'
        )

    command_html = "<br>".join(formatted_lines)

    return f"""
    <div style="margin: 0px 0 0 0;">
        <div style="background-color: #f5f5f5; padding: 6px 8px; border-radius: 4px; font-family: 'Consolas', monospace; font-size: 0.9em;">
        {command_html}
        </div>
    """


def format_output_html(entry):
    """Format the output part of a console entry as HTML."""
    if not entry.output:
        return "</div>"

    escaped_output = (
        entry.output.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace(" ", "&nbsp;")
        .replace("\n", "<br>")
    )

    return f"""
    <div style="background-color: #ffffff; padding: 6px 8px; border-left: 3px solid {"#ff3860" if entry.is_error else "#2196F3"}; margin-top: 2px; font-family: 'Consolas', monospace; font-size: 0.9em; {"color: #ff3860;" if entry.is_error else ""}">
    {escaped_output}
    </div>
    </div>
    """


@solara.component
def ConsoleInput(on_submit, on_up, on_down):
    """A solara component for handling console input."""
    input_text, set_input_text = solara.use_state("")

    def handle_submit(*ignore_args):
        """Handle submit."""
        on_submit(input_text, set_input_text)

    def handle_up(*ignore_args):
        """Handle up."""
        on_up(input_text, set_input_text)

    def handle_down(*ignore_args):
        """Handle down."""
        on_down(set_input_text)

    input_elem = solara.v.TextField(
        v_model=input_text,
        on_v_model=set_input_text,
        flat=True,
        hide_details=True,
        dense=True,
        height="auto",
        background_color="transparent",
        style_="font-family: monospace; border: none; box-shadow: none; padding: 0; margin: 0; background-color: transparent; color: #000; flex-grow: 1;",
        placeholder="",
        solo=False,
        filled=False,
        outlined=False,
        id="console-input",
        attributes={
            "spellcheck": "false",
            "autocomplete": "off",
        },
    )

    # Bind key events with the input element
    use_change(input_elem, handle_submit, update_events=["keypress.enter"])
    use_change(input_elem, handle_up, update_events=["keyup.38"])  # 38 -> Up arrow
    use_change(
        input_elem, handle_down, update_events=["keydown.40"]
    )  # 40 -> Down arrow

    return input_elem


@solara.component
def CommandConsole(model=None, additional_imports=None):
    """A solara component for executing Python code interactively in the browser."""
    # Initialize state for the console manager
    console_ref = solara.use_ref(None)
    if console_ref.current is None:
        console_ref.current = ConsoleManager(
            model=model, additional_imports=additional_imports
        )

    # State to trigger re-renders
    refresh, set_refresh = solara.use_state(0)

    def handle_code_execution(code, set_input_text):
        """Handle code execution."""
        console_ref.current.execute_code(code, set_input_text)
        set_refresh(refresh + 1)

    def handle_up(current_text, set_input_text):
        """Handle up."""
        console_ref.current.prev_command(current_text, set_input_text)
        set_refresh(refresh + 1)

    def handle_down(set_input_text):
        """Handle down."""
        console_ref.current.next_command(set_input_text)
        set_refresh(refresh + 1)

    with solara.Column(
        style={
            "height": "300px",
            "overflow-y": "auto",
            "gap": "0px",
            "box-shadow": "inset 0 0 10px rgba(0,0,0,0.1)",
            "border": "3px solid #e0e0e0",
            "border-radius": "6px",
            "padding": "8px",
        }
    ):
        console_entries = console_ref.current.get_entries()

        # Display history entries with auto-scrolling
        with solara.v.ScrollYTransition(group=True):
            for entry in console_entries:
                with solara.Div():
                    command_html = format_command_html(entry)
                    output_html = format_output_html(entry)
                    solara.Markdown(command_html + output_html)

        # Input row that adapts to content above it
        with solara.Row(
            style={"align-items": "center", "margin": "0", "width": "94.5%"}
        ):
            solara.Text(">>> ", style={"color": "#0066cc"})
            ConsoleInput(
                on_submit=handle_code_execution, on_up=handle_up, on_down=handle_down
            )

    solara.Markdown(
        "*Type 'tips' for usage instructions.*",
        style="font-size: 0.8em; color: #666;",
    )
