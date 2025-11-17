"""
common CLI utilities for interactive API testing tools.

provides reusable functions for building interactive command-line interfaces
to test various APIs in the fact-checking pipeline.
"""

import sys
from typing import Optional, Callable, Any, Dict, List
import json


# ===== TEXT FORMATTING =====

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str) -> None:
    """print a bold header with separators."""
    separator = "=" * 80
    print(f"\n{Colors.BOLD}{Colors.CYAN}{separator}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{separator}{Colors.END}\n")


def print_section(text: str) -> None:
    """print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'─' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─' * 80}{Colors.END}")


def print_success(text: str) -> None:
    """print success message in green."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str) -> None:
    """print error message in red."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str) -> None:
    """print warning message in yellow."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str) -> None:
    """print info message in cyan."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.END}")


def print_json(data: Any, indent: int = 2) -> None:
    """print formatted JSON with syntax highlighting."""
    formatted = json.dumps(data, indent=indent, ensure_ascii=False)
    print(formatted)


def print_dict_table(data: Dict[str, Any], title: Optional[str] = None) -> None:
    """print a dictionary as a formatted table."""
    if title:
        print(f"\n{Colors.BOLD}{title}{Colors.END}")

    max_key_length = max(len(str(k)) for k in data.keys()) if data else 0

    for key, value in data.items():
        key_str = str(key).ljust(max_key_length)
        value_str = str(value)

        # truncate very long values
        if len(value_str) > 100:
            value_str = value_str[:97] + "..."

        print(f"  {Colors.BOLD}{key_str}{Colors.END}: {value_str}")


# ===== USER INPUT =====

def prompt_input(message: str, default: Optional[str] = None) -> str:
    """prompt user for text input with optional default."""
    if default:
        prompt = f"{Colors.BOLD}{message}{Colors.END} [{default}]: "
    else:
        prompt = f"{Colors.BOLD}{message}{Colors.END}: "

    user_input = input(prompt).strip()

    if not user_input and default:
        return default

    return user_input


def prompt_yes_no(message: str, default: bool = True) -> bool:
    """prompt user for yes/no confirmation."""
    default_str = "Y/n" if default else "y/N"
    prompt = f"{Colors.BOLD}{message}{Colors.END} [{default_str}]: "

    while True:
        response = input(prompt).strip().lower()

        if not response:
            return default

        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print_error("Please enter 'y' or 'n'")


def prompt_choice(message: str, choices: List[str], default: Optional[str] = None) -> str:
    """prompt user to select from a list of choices."""
    print(f"\n{Colors.BOLD}{message}{Colors.END}")

    for i, choice in enumerate(choices, 1):
        default_marker = " (default)" if choice == default else ""
        print(f"  {i}. {choice}{default_marker}")

    while True:
        response = input(f"{Colors.BOLD}Enter choice [1-{len(choices)}]{Colors.END}: ").strip()

        if not response and default:
            return default

        try:
            index = int(response) - 1
            if 0 <= index < len(choices):
                return choices[index]
            else:
                print_error(f"Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print_error("Please enter a valid number")


def prompt_multiline(message: str, end_marker: str = "END") -> str:
    """prompt user for multiline text input."""
    print(f"{Colors.BOLD}{message}{Colors.END}")
    print(f"{Colors.YELLOW}(Enter '{end_marker}' on a new line to finish){Colors.END}")

    lines = []
    while True:
        line = input()
        if line.strip() == end_marker:
            break
        lines.append(line)

    return "\n".join(lines)


# ===== INTERACTIVE MENU =====

class Menu:
    """interactive menu system for CLI tools."""

    def __init__(self, title: str):
        self.title = title
        self.options: Dict[str, Callable[[], Any]] = {}
        self.option_labels: List[str] = []

    def add_option(self, label: str, handler: Callable[[], Any]) -> None:
        """add a menu option with its handler function."""
        self.options[label] = handler
        self.option_labels.append(label)

    def add_separator(self) -> None:
        """add a visual separator in the menu."""
        self.option_labels.append("---")

    def display(self) -> None:
        """display the menu options."""
        print_header(self.title)

        option_number = 1
        for label in self.option_labels:
            if label == "---":
                print(f"  {Colors.BLUE}{'─' * 40}{Colors.END}")
            else:
                print(f"  {option_number}. {label}")
                option_number += 1

        print(f"  0. Exit")

    def run(self) -> None:
        """run the interactive menu loop."""
        while True:
            self.display()

            # map displayed numbers to actual options
            valid_options = [label for label in self.option_labels if label != "---"]

            try:
                choice = input(f"\n{Colors.BOLD}Select option{Colors.END}: ").strip()

                if choice == "0":
                    print_info("Exiting...")
                    break

                choice_num = int(choice)
                if 1 <= choice_num <= len(valid_options):
                    selected_label = valid_options[choice_num - 1]
                    handler = self.options[selected_label]

                    try:
                        handler()
                    except KeyboardInterrupt:
                        print_warning("\nOperation cancelled")
                    except Exception as e:
                        print_error(f"Error: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print_error(f"Invalid choice. Please enter 0-{len(valid_options)}")

            except ValueError:
                print_error("Please enter a valid number")
            except KeyboardInterrupt:
                print_info("\nExiting...")
                break


# ===== RESULT FORMATTING =====

def print_api_response(
    endpoint: str,
    status_code: int,
    headers: Dict[str, str],
    data: Any,
    show_full_response: bool = True
) -> None:
    """print formatted API response."""
    print_section(f"API Response: {endpoint}")

    # status
    if status_code == 200:
        print_success(f"Status: {status_code} OK")
    else:
        print_error(f"Status: {status_code}")

    # headers (selected important ones)
    important_headers = {
        'content-type': headers.get('content-type'),
        'content-length': headers.get('content-length'),
        'date': headers.get('date'),
    }
    print_dict_table({k: v for k, v in important_headers.items() if v}, "Important Headers")

    # response data
    if show_full_response:
        print(f"\n{Colors.BOLD}Response Body:{Colors.END}")
        print_json(data)
    else:
        print_info("Response body hidden (use show_full=True to display)")


def print_citation_list(citations: List[Any]) -> None:
    """print a formatted list of citations."""
    if not citations:
        print_warning("No citations found")
        return

    print_success(f"Found {len(citations)} citation(s):")

    for i, citation in enumerate(citations, 1):
        print(f"\n{Colors.BOLD}Citation {i}:{Colors.END}")
        print(f"  Title: {citation.title}")
        print(f"  Publisher: {citation.publisher}")
        print(f"  Rating: {Colors.BOLD}{citation.rating or 'N/A'}{Colors.END}")
        if citation.rating_comment:
            print(f"  Comment: {citation.rating_comment}")
        print(f"  URL: {Colors.CYAN}{citation.url}{Colors.END}")
        if citation.date:
            print(f"  Date: {citation.date}")


# ===== LOADING INDICATORS =====

def with_spinner(func: Callable[[], Any], message: str = "Processing...") -> Any:
    """execute a function with a loading spinner."""
    import itertools
    import threading
    import time

    spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    stop_spinner = False
    result = None
    exception = None

    def spin():
        while not stop_spinner:
            sys.stdout.write(f'\r{Colors.CYAN}{next(spinner)} {message}{Colors.END}')
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(message) + 5) + '\r')
        sys.stdout.flush()

    def run_func():
        nonlocal result, exception
        try:
            result = func()
        except Exception as e:
            exception = e

    spinner_thread = threading.Thread(target=spin)
    func_thread = threading.Thread(target=run_func)

    spinner_thread.start()
    func_thread.start()

    func_thread.join()
    stop_spinner = True
    spinner_thread.join()

    if exception:
        raise exception

    return result


# ===== ERROR HANDLING =====

def handle_api_error(e: Exception, api_name: str) -> None:
    """handle and format API errors."""
    print_error(f"{api_name} Error")

    # try to extract useful error info
    if hasattr(e, 'response'):
        print(f"  Status Code: {e.response.status_code}")
        print(f"  Response: {e.response.text[:200]}")
    else:
        print(f"  Error: {str(e)}")

    # show full traceback in debug mode
    import traceback
    print(f"\n{Colors.YELLOW}Full traceback:{Colors.END}")
    traceback.print_exc()
