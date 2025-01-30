import requests
from bs4 import BeautifulSoup
import re

def fetch_tab_content(url):
    try:
        # Fetch the webpage content
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract title
        title = soup.title.string if soup.title else "No title available"
        
        # Extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = meta_desc["content"] if meta_desc else "No meta description available"
        
        # Extract the first 200 words from visible text
        texts = soup.find_all(text=True)
        visible_texts = filter(tag_visible, texts)
        full_text = " ".join(t.strip() for t in visible_texts)
        snippet = " ".join(full_text.split()[:200])  # Get first 200 words

        return {
            "title": title,
            "description": description,
            "snippet": snippet
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"Error fetching URL: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

def tag_visible(element):
    """
    Determine if an HTML element is visible to the user.
    """
    if element.parent.name in ['style', 'script', 'head', 'meta', '[document]']:
        return False
    if isinstance(element, str):
        return True
    return False

# Example usage
if __name__ == "__main__":
    url = input("Enter a URL to fetch context from: ").strip()
    context = fetch_tab_content(url)
    print("\nTab Context:\n", context)
