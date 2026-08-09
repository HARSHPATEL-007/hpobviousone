import requests
from bs4 import BeautifulSoup

def scrape_web_content(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    # Example: Extract all text from paragraphs
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    return paragraphs

if __name__ == "__main__":
    url = "https://vscode-f67015b1-67d8-47a8-8f35-f5e337c39a6e.preview.emergentagent.com/"
    paragraphs = scrape_web_content(url)
    for i, text in enumerate(paragraphs, 1):
        print(f"Paragraph {i}: {text}\n")
