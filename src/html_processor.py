"""
HTML Processing and Text Extraction Utility.

This module is responsible for fetching and parsing HTML content from local files or URLs.
It uses the BeautifulSoup library to extract clean, readable text from the HTML structure,
optionally preserving table formats.

The main function, `extract_text_from_html`, is used by the extraction pipeline to prepare
HTML content for analysis by the language model.
"""

from bs4 import BeautifulSoup
import requests
from logger import setup_logger

logger = setup_logger(__name__)

def _table_to_markdown(table):
    """Converts an HTML table to markdown format."""
    rows = []
    for tr in table.find_all('tr'):
        cells = []
        for td in tr.find_all(['td', 'th']):
            cell_text = ' '.join(td.stripped_strings)
            cells.append(cell_text)
        if cells:
            rows.append(cells)
    
    if not rows:
        return ""
    
    # Create markdown table
    markdown = []
    # Header row
    if rows:
        markdown.append('| ' + ' | '.join(rows[0]) + ' |')
        markdown.append('| ' + ' | '.join(['---'] * len(rows[0])) + ' |')
        # Data rows
        for row in rows[1:]:
            # Pad row if needed
            while len(row) < len(rows[0]):
                row.append('')
            markdown.append('| ' + ' | '.join(row) + ' |')
    
    return '\n'.join(markdown)

def extract_text_from_html(html_path_or_url: str, preserve_tables: bool = True) -> str:
    """
    Extracts the main text content from an HTML file or URL.
    
    Args:
        html_path_or_url: The path to the HTML file or a URL.
        preserve_tables: If True, converts tables to markdown format. Default is True.
    
    Returns:
        A string containing the extracted text with tables preserved if requested.
    """
    html_content = ""
    if html_path_or_url.startswith('http'):
        try:
            response = requests.get(html_path_or_url)
            response.raise_for_status()  # Raise an exception for bad status codes
            html_content = response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL {html_path_or_url}: {e}")
            return ""
    else:
        try:
            with open(html_path_or_url, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except FileNotFoundError:
            logger.error(f"File not found at {html_path_or_url}")
            return ""
        except Exception as e:
            logger.error(f"Error reading file {html_path_or_url}: {e}")
            return ""

    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Extract tables first if preserving them
        if preserve_tables:
            tables = soup.find_all('table')
            for table in tables:
                # Replace table with a placeholder that includes markdown
                markdown_table = _table_to_markdown(table)
                if markdown_table:
                    # Create a new tag to hold the markdown table
                    placeholder = soup.new_tag('div')
                    placeholder.string = f"\n\n[TABLE]\n{markdown_table}\n[/TABLE]\n\n"
                    table.replace_with(placeholder)

        # Get text (now with tables as markdown)
        text = soup.get_text()

        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        logger.error(f"Error processing HTML content: {e}")
        return ""
