from crewai import Agent, LLM
import os

from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools


class TripAgents:

    def __init__(self):
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # CrewAI's LLM class with Ollama provider
        self.llm = LLM(
            model=f"ollama/{ollama_model}",
            base_url=ollama_base_url
        )


    def city_selection_agent(self):
        return Agent(
            role='City Selection Expert',
            goal=("Select the best city based on weather, season, and prices\n"
            "Rules:\n"
            "- You may use at most ONE tool.\n"
            "- After receiving tool results, you MUST provide the final choice of city.\n"
            "- The final answer MUST NOT contain 'Action', 'Thought', or tool calls.\n"
            "- The final answer must only contain the city of your choice and explanation."),
            backstory='An expert in analyzing travel data to pick ideal destinations',
            tools=[
                SearchTools.search_internet,
                BrowserTools.scrape_and_summarize_website,
            ],
            llm=self.llm,
            max_iter=3,
            allow_delegation=False,
            verbose=True
        )

    def local_expert(self):
        return Agent(
            role='Local Expert at this city',
            goal=("Provide the BEST insights about the selected city\n"
            "Rules:\n"
            "- You may use at most ONE tool.\n"
            "- After receiving tool results, you MUST provide the final list of insights.\n"
            "- The final answer MUST NOT contain 'Action', 'Thought', or tool calls.\n"
            "- The final answer must only contain the completed list of insights."),
            backstory=("""A knowledgeable local guide with extensive information
            about the city, its attractions and customs"""),
            tools=[
                SearchTools.search_internet,
                BrowserTools.scrape_and_summarize_website,
            ],
            llm=self.llm,
            max_iter=3,
            allow_delegation=False,
            verbose=True
        )

    def travel_concierge(self):
        return Agent(
            role='Amazing Travel Concierge',
            goal=("Create the most amazing travel itineraries with budget and packing suggestions for the city"
            "Rules:\n"
            "- You may use at most ONE tool.\n"
            "- After receiving tool results, you MUST provide the final itinerary.\n"
            "- The final answer MUST NOT contain 'Action', 'Thought', or tool calls.\n"
            "- The final answer must only contain the completed itinerary."),
            backstory=("""Specialist in travel planning and logistics with 
            decades of experience"""),
            tools=[
                SearchTools.search_internet,
                BrowserTools.scrape_and_summarize_website,
                CalculatorTools.calculate,
            ],
            llm=self.llm,
            max_iter=3,
            allow_delegation=False,
            verbose=True
        )
