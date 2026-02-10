import requests
from crewai import Agent, Task
from crewai.tools import tool
from bs4 import BeautifulSoup


class BrowserTools():

  @tool("Scrape website content")
  def scrape_and_summarize_website(website: str) -> str:
    """Useful to scrape and summarize a website content.
    
    Args:
        website: The URL of the website to scrape and summarize
    
    Returns:
        Summarized content from the website
    """
    try:
      headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
      }
      response = requests.get(website, headers=headers, timeout=60)
      response.raise_for_status()
      html_text = response.text
    except Exception as e:
      return f"Error fetching website content: {str(e)}"

    soup = BeautifulSoup(html_text, 'html.parser')
    
    # Remove script and style elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header']):
      element.decompose()
    
    # Extract text content
    text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'span', 'div'])
    content = "\n\n".join([el.get_text(strip=True) for el in text_elements if el.get_text(strip=True)])
    content = [content[i:i + 8000] for i in range(0, len(content), 8000)]
    summaries = []
    for chunk in content:
      agent = Agent(
          role='Principal Researcher',
          goal=
          'Do amazing researches and summaries based on the content you are working with',
          backstory=
          "You're a Principal Researcher at a big company and you need to do a research about a given topic.",
          allow_delegation=False)
      task = Task(
          agent=agent,
          description=
          f'Analyze and summarize the content bellow, make sure to include the most relevant information in the summary, return only the summary nothing else.\n\nCONTENT\n----------\n{chunk}',
          expected_output="A concise markdown summary capturing key facts, figures, entities, and links (if present)."
      )
      try:
        summary = task.execute()
      except Exception:
        # Fallback summarization if LLM execution fails
        sample = (chunk[:1000] + '...') if len(chunk) > 1000 else chunk
        summary = f"Fallback summary (no LLM available):\n\n{sample}"
      summaries.append(summary)
    return "\n\n".join(summaries)
