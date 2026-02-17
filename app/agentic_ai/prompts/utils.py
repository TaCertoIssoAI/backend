from datetime import datetime, timezone

# date format for fact-checking context: DD-MM-YYYY
DATE_FORMAT = "%d-%m-%Y"

def get_current_date() -> str:
    """
    Returns the current date in DD-MM-YYYY format using UTC timezone.

    Returns:
        Formatted date string (e.g., "08-12-2024")
    """
    now = datetime.now(timezone.utc)
    return now.strftime(DATE_FORMAT)