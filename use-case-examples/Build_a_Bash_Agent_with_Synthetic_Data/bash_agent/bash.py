"""
Bash command execution tool with security features.

Aligned with NVIDIA GenerativeAIExamples bash_computer_use_agent/bash.py
"""

from typing import Any, Dict, List
import re
import shlex
import subprocess

from config import Config


class Bash:
    """
    An implementation of a tool that executes bash commands and keeps track of the working directory.

    Security features:
    - Command allowlist to restrict executable commands
    - Injection pattern detection (backticks, $() substitution)
    - Working directory tracking
    """

    def __init__(self, config: Config):
        self.config = config
        # The current working directory (tracked and updated throughout the session)
        self.cwd = config.root_dir
        # Set the initial working directory
        self._run_bash_command(f"cd {self.cwd}")

    def exec_bash_command(self, cmd: str) -> Dict[str, str]:
        """
        Execute the bash command after checking the allowlist.

        Args:
            cmd: The bash command to execute

        Returns:
            Dictionary with stdout, stderr, and current working directory
        """
        if cmd:
            # Prevent command injection via backticks or $. This blocks variables too.
            if re.search(r"[`$]", cmd):
                return {"error": "Command injection patterns are not allowed."}

            # Check the allowlist
            for cmd_part in self._split_commands(cmd):
                if cmd_part not in self.config.allowed_commands:
                    return {
                        "error": f"Command '{cmd_part}' is not in the allowlist. "
                        f"Allowed commands: {self.config.allowed_commands}"
                    }

            return self._run_bash_command(cmd)

        return {"error": "No command was provided"}

    def to_json_schema(self) -> Dict[str, Any]:
        """
        Convert the function signature to a JSON schema for LLM tool calling.

        Returns:
            OpenAI-compatible tool schema
        """
        return {
            "type": "function",
            "function": {
                "name": "exec_bash_command",
                "description": "Execute a bash command and return stdout/stderr and the working directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "The bash command to execute"
                        }
                    },
                    "required": ["cmd"],
                },
            },
        }

    def _split_commands(self, cmd_str: str) -> List[str]:
        """
        Split a command string into individual commands, without the parameters.

        Handles command chaining with ;, &, and |.

        Args:
            cmd_str: The command string to split

        Returns:
            List of command names (without arguments)
        """
        parts = re.split(r'[;&|]+', cmd_str)
        commands = []

        for part in parts:
            try:
                tokens = shlex.split(part.strip())
                if tokens:
                    commands.append(tokens[0])
            except ValueError:
                # Handle malformed shell strings
                stripped = part.strip().split()[0] if part.strip() else ""
                if stripped:
                    commands.append(stripped)

        return commands

    def _run_bash_command(self, cmd: str) -> Dict[str, str]:
        """
        Runs the bash command and catches exceptions (if any).

        Args:
            cmd: The command to execute

        Returns:
            Dictionary with stdout, stderr, and cwd
        """
        stdout = ""
        stderr = ""
        new_cwd = self.cwd

        try:
            # Wrap the command so we can keep track of the working directory
            wrapped = f"{cmd};echo __END__;pwd"
            result = subprocess.run(
                wrapped,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                executable="/bin/bash",
                timeout=30,  # Prevent hanging commands
            )
            stderr = result.stderr
            # Find the separator marker
            split = result.stdout.split("__END__")
            stdout = split[0].strip()

            # If no output/error at all, inform that the call was successful
            if not stdout and not stderr:
                stdout = "Command executed successfully, without any output."

            # Get the new working directory, and change it
            new_cwd = split[-1].strip()
            self.cwd = new_cwd
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = "Command timed out after 30 seconds."
        except Exception as e:
            stdout = ""
            stderr = str(e)

        return {
            "stdout": stdout,
            "stderr": stderr,
            "cwd": new_cwd,
        }
