"""
HTML Processing.

Fetches and parses HTML content from URLs or files, converting tables
to markdown to preserve structure for LLM analysis.
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
        # Clean up cell text to prevent markdown table breakage
        def clean_cell(text):
            return text.replace('|', '\\|').replace('\n', ' ').strip()

        header = [clean_cell(cell) for cell in rows[0]]
        markdown.append('| ' + ' | '.join(header) + ' |')
        markdown.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
        
        # Data rows
        for row in rows[1:]:
            # Pad row if needed
            while len(row) < len(header):
                row.append('')
            
            cleaned_row = [clean_cell(cell) for cell in row]
            markdown.append('| ' + ' | '.join(cleaned_row) + ' |')
    
    return '\n'.join(markdown)

def extract_text_from_html(html_path_or_url: str, preserve_tables: bool = True, raw_html: bool = False) -> str:
    """
    Extracts the main text content from an HTML file, URL, or raw HTML string.
    
    Args:
        html_path_or_url: A URL, file path, or raw HTML string.
        preserve_tables: If True, converts tables to markdown format. Default is True.
        raw_html: If True, treats the first argument as a raw HTML string directly.
    
    Returns:
        A string containing the extracted text with tables preserved if requested.
    """
    if raw_html:
        html_content = html_path_or_url
    else:
        try:
            if html_path_or_url.startswith('http'):
                response = requests.get(html_path_or_url, timeout=15)
                response.raise_for_status()
                html_content = response.text
            else:
                with open(html_path_or_url, 'r', encoding='utf-8') as f:
                    html_content = f.read()
        except Exception as e:
            logger.error(f"Error reading HTML from {html_path_or_url}: {e}")
            return ""

    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Target the specific content div for MFN if it exists
        # Based on the user's provided XPath: /html/body/div[3]/div[3]/div[1]/div[2]/div/div[1]/a
        # This corresponds to the 'full-item' container and its 'title' div.
        
        # 1. Try to find the headline specifically
        headline_div = soup.find('div', class_='title')
        headline_text = ""
        if headline_div:
            headline_text = headline_div.get_text(strip=True)

        # 2. Try to find the main content container
        main_content = soup.find('div', class_='full-item') or soup.find('div', class_='content') or soup.find('article')
        
        if main_content:
            # If we found a specific content container, use only that
            soup = BeautifulSoup(str(main_content), 'html.parser')
        else:
            # Fallback: Remove script, style, and navigation/header/footer elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

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
        # Use separator to prevent merging text from different tags
        text = soup.get_text(separator="\n")

        # Clean up the text:
        # 1. Strip whitespace from each line
        # 2. Filter out empty lines
        # 3. Join with single newlines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # If we found a headline earlier, make sure it's at the very top
        if headline_text and (not lines or headline_text not in lines[0]):
            lines.insert(0, headline_text)
            
        text = '\n'.join(lines)

        return text
    except Exception as e:
        logger.error(f"Error processing HTML content: {e}")
        return ""
