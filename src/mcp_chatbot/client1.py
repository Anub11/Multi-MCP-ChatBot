import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = Path(__file__).resolve().with_name("server.py")

load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------
# 2. Local MCP server configuration
# --------------------------------------------------

LOCAL_SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "--project",
            str(PROJECT_ROOT),
            "fastmcp",
            "run",
            str(SERVER_PATH),
        ],
    },
    "manim": {
        "transport": "stdio",
        "command": "python",
        "args": [
            r"D:\LLMOPS\MCP\MCP Project\manim-mcp-server\src\manim_server.py"
        ],
    },
}


# --------------------------------------------------
# 3. Connect to local MCP servers
# --------------------------------------------------

async def main():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            f"GROQ_API_KEY is missing in {PROJECT_ROOT / '.env'}"
        )

    client = MultiServerMCPClient(LOCAL_SERVERS)

    tools = await client.get_tools()

    print("\nAvailable MCP tools:")

    for tool in tools:
        print(f"- {tool.name}")

    # --------------------------------------------------
    # 4. Create a tool lookup dictionary
    # --------------------------------------------------

    named_tools = {
        tool.name: tool
        for tool in tools
    }

    # --------------------------------------------------
    # 5. Initialize Groq
    # --------------------------------------------------

    llm = ChatGroq(
        model=os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        ),
        api_key=api_key,
        temperature=0,
    )

    llm_with_tools = llm.bind_tools(tools)

    # --------------------------------------------------
    # 6. User prompt
    # --------------------------------------------------

    prompt = (
        "Draw a triangle using manim tool."
    )

    response = await llm_with_tools.ainvoke(
        [HumanMessage(content=prompt)]
    )

    # --------------------------------------------------
    # 7. Execute requested tools
    # --------------------------------------------------

    if not response.tool_calls:
        print("\nLLM Reply:")
        print(response.content)
        return

    tool_messages = []

    for tool_call in response.tool_calls:

        selected_tool = tool_call["name"]
        selected_tool_args = tool_call.get("args") or {}
        selected_tool_id = tool_call["id"]

        print(f"\nCalling tool: {selected_tool}")
        print(f"Arguments: {selected_tool_args}")

        try:
            tool_result = await named_tools[
                selected_tool
            ].ainvoke(selected_tool_args)

            if isinstance(tool_result, str):
                content = tool_result
            else:
                content = json.dumps(
                    tool_result,
                    ensure_ascii=False,
                    default=str,
                )

        except Exception as exc:
            content = json.dumps({
                "error": str(exc)
            })

        print("Tool result:", content)

        tool_messages.append(
            ToolMessage(
                content=content,
                tool_call_id=selected_tool_id,
            )
        )

    # --------------------------------------------------
    # 8. Send results back to Groq
    # --------------------------------------------------

    final_messages = [
        HumanMessage(content=prompt),
        response,
        *tool_messages,
    ]

    final_response = await llm_with_tools.ainvoke(
        final_messages
    )

    print("\nFinal response:")
    print(final_response.content)


if __name__ == "__main__":
    asyncio.run(main())