from __future__ import annotations as _annotations

import asyncio
import os
from typing import Literal, Optional
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.usage import UsageLimits
from pydantic_ai.messages import ModelMessage
from mcp import ClientSession
from contextlib import AsyncExitStack

import logfire

from dotenv import load_dotenv

load_dotenv()


def get_llm_endpoint(provider: Literal["LOCAL", "GEMINI"] = "LOCAL"):
    """Returns the complete LLM API endpoint URL"""
    base_url = os.getenv(f"LLM_BASE_URL_{provider}", "")
    return base_url


def get_model_name(provider: Literal["LOCAL", "GEMINI"] = "LOCAL"):
    """Returns the model name to use for API requests"""
    return os.getenv(f"LLM_MODEL_NAME_{provider}", "")


def get_llm_api_key(provider: Literal["LOCAL", "GEMINI"] = "LOCAL"):
    """Returns the llm api keyto use for API requests"""
    return os.getenv(f"LLM_API_KEY_{provider}", "")


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.state: list[ModelMessage] = []
        self.tools: list[MCPServerStdio] = []
        self.agent: Optional[Agent] = None

    async def connect_to_server(self):
        # setup MCP servers
        weather = MCPServerStdio(
            "python",
            args=["mcp-weather/weather.py"],
        )

        obsidian = MCPServerStdio(
            "uvx",
            args=["mcp-obsidian"],
            env={
                "OBSIDIAN_API_KEY": os.getenv("OBSIDIAN_API_KEY"),
                "OBSIDIAN_HOST": os.getenv("OBSIDIAN_HOST"),
            },
        )

        filesystem = MCPServerStdio(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "./"],
        )

        everything = MCPServerStdio(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
        )
        self.tools.extend([weather, obsidian, filesystem, everything])

    def setup_agent(self):
        # Create nodes for different agents
        model_local = OpenAIModel(
            model_name=get_model_name("LOCAL"),
            provider=OpenAIProvider(
                base_url=get_llm_endpoint("LOCAL"),
                api_key=get_llm_api_key("LOCAL"),
            ),
        )
        _ = Agent(
            model_local,
            mcp_servers=self.tools,
            output_type=str,
            model_settings={
                "frequency_penalty": 8.0,
                "temperature": 0.1,
            },
            end_strategy="early",
            system_prompt=(
                "If the question is not related to the mcp servers, answer it using your own knowledge. "
                "If the question is related to the mcp servers, use the functions to answer it. "
                "Use the answers from the mcp servers to answer the user's question. "
                "If 'isError' is False for the tool response, use the content from the tool response to provide an answer to the user. "
                "You get answers from the mcp servers. Use the answers from the mcp servers to answer the user's question. "
                "Provide a long answer at the end."
            ),
            instrument=True,
        )

        model_gemini = GeminiModel(
            model_name=get_model_name("GEMINI"),
            provider=GoogleGLAProvider(
                api_key=get_llm_api_key("GEMINI"),
            ),
        )

        agent_gemini = Agent(
            model_gemini,
            mcp_servers=self.tools,
            output_type=str,
            instrument=True,
            system_prompt=(
                "Be a helpful assistant. "
                "Focus mostly on the last user's question."
                "If the question is not related to the mcp servers, answer it using your own knowledge. "
                "If the question is related to the mcp servers, use the functions to answer it. "
                "If someone asks you about the weather forecast of a city, figure out the latidude and longitude of the city yourself. "
            ),
        )

        self.agent = agent_gemini

    async def process_query(self, user_message: str) -> str:
        self.setup_agent()

        if not self.agent:
            raise RuntimeError("Agent not initialized")

        async with self.agent.run_mcp_servers():
            try:
                async with self.agent.run_stream(
                    user_message,
                    message_history=self.state,
                    usage_limits=UsageLimits(request_limit=15),
                ) as result:
                    self.state = result.all_messages()
                    return await result.get_output()
            except Exception as e:
                print(f"Error: {e}")
                return ""

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nAgent Graph Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()


# uv run client.py
if __name__ == "__main__":
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN", ""))
    asyncio.run(main())
