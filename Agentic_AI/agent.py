from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv

load_dotenv()


def build_agent():
    return Agent(
        model=OpenRouter(
            id="openai/gpt-4o",
            api_key=None,
            base_url="https://openrouter.ai/api/v1"
        ),
        tools=[DuckDuckGoTools()],
        markdown=True,
        instructions="You are a helpful assistant that can answer questions and provide information on a wide range of topics. Please provide clear and concise responses.",
        add_datetime_to_context=True,
    )


agent = build_agent()

agent.print_response("What is the capital of France?")