"""
HTML Processing Module.

This module provides functions to extract clean, readable text content from HTML files. It uses BeautifulSoup to parse the HTML and heuristics to remove common non-content elements like scripts, styles, and navigation bars.
"""

from bs4 import BeautifulSoup
import requests

def extract_text_from_html(html_path_or_url: str) -> str:
    """
    Extracts the main text content from an HTML file or URL.

    Args:
        html_path_or_url: The path to the HTML file or a URL.

    Returns:
        A string containing the extracted text.
    """
    html_content = ""
    if html_path_or_url.startswith('http'):
        try:
            response = requests.get(html_path_or_url)
            response.raise_for_status()  # Raise an exception for bad status codes
            html_content = response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {html_path_or_url}: {e}")
            return ""
    else:
        try:
            with open(html_path_or_url, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except FileNotFoundError:
            print(f"Error: File not found at {html_path_or_url}")
            return ""
        except Exception as e:
            print(f"Error reading file {html_path_or_url}: {e}")
            return ""

    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        print(f"Error processing HTML content: {e}")
        return ""
