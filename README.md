# MCP project
Obsidian MCP server setup is from here: https://github.com/MarkusPfundstein/mcp-obsidian
Weather MCP server setup is from Anthropic documentation.

Find more here: https://github.com/punkpeye/awesome-mcp-servers
MCP documentation: https://modelcontextprotocol.io/introduction

1. Setup

```shell
uv sync
```

```conf
# .env
LLM_MODEL_NAME_GEMINI=gemini-2.0-flash
LLM_API_KEY_GEMINI=xxx # needs google api key from https://aistudio.google.com

LOGFIRE_TOKEN=xxx # needs logfire token
OBSIDIAN_API_KEY=xxx # needs obsidian REST Api plugin and token
OBSIDIAN_HOST=127.0.0.1
```

2. Start chat in console
```shell
uv run client_chat.py
```

Example user prompts:

```
What is the weather in New York?

What files do I have in my vault?
Create a file x.md with the content of the forecast in San Francisco. 
```


Check interactions in logfire: https://logfire-eu.pydantic.dev
Check Google API usage here: https://aistudio.google.com


## Use it with local running LLM
Use docker:
```shell
docker model pull ai/qwen3
docker desktop enable model-runner --tcp 12434
```

Add the following to your `.env` file:

```conf
LLM_BASE_URL_LOCAL=http://localhost:12434/engines/llama.cpp/v1
LLM_MODEL_NAME_LOCAL=ai/qwen3
LLM_API_KEY_LOCAL=na
```

Beware that local models are not as powerful and will end up likely in a worse behaviour than the bigger models.

## Next steps
- Langgraph or pydantic-ai-graph to orchestrate stuff
- Agentic Framework to orchestrate between multiple Agents
