import asyncio
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = Path(__file__).resolve().with_name("server.py")

load_dotenv(PROJECT_ROOT / ".env")


# ==========================================================
# 2. MCP SERVER CONFIGURATION
# ==========================================================

MCP_SERVERS = {
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
    "weather": {
        "transport": "streamable_http",
        "url": "https://open-meteo.caseyjhand.com/mcp",
    },
}


# ==========================================================
# 3. LLM CONFIGURATION
# ==========================================================

def initialize_llm(tools):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"OPENROUTER_API_KEY is missing from {PROJECT_ROOT / '.env'}"
        )

    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b"),
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_retries=2,
    )
    return llm.bind_tools(tools)


# ==========================================================
# 4. MCP TOOL DISCOVERY
# ==========================================================

async def discover_mcp_tools():
    """Discover tools. Clients are deliberately not stored across event loops."""
    all_tools = []
    named_tools = {}
    tool_to_server = {}
    server_tool_names = {}

    for server_name, server_config in MCP_SERVERS.items():
        client = MultiServerMCPClient({server_name: server_config})
        tools = await client.get_tools()
        server_tool_names[server_name] = []

        for tool in tools:
            if tool.name in named_tools:
                raise ValueError(
                    f"Duplicate MCP tool name '{tool.name}'. "
                    "Tool names must be unique across the configured servers."
                )

            named_tools[tool.name] = tool
            tool_to_server[tool.name] = server_name
            server_tool_names[server_name].append(tool.name)
            all_tools.append(tool)

    if not all_tools:
        raise RuntimeError("No MCP tools were discovered.")

    return all_tools, named_tools, tool_to_server, server_tool_names


# ==========================================================
# 5. CONVERSATION AND TOOL EXECUTION
# ==========================================================

def convert_history(history):
    messages = []
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            # Only include plain assistant text in history.
            # Prior tool-call AI messages are not reconstructed here.
            messages.append(AIMessage(content=item["content"]))
    return messages


async def run_chat_turn(user_prompt, conversation_history):
    """
    Do discovery, LLM binding, tool execution, and final response in ONE
    asyncio.run() call. This avoids reusing loop-bound resources after a
    previous event loop has closed.
    """
    tools, named_tools, tool_to_server, server_tool_names = (
        await discover_mcp_tools()
    )
    llm_with_tools = initialize_llm(tools)

    messages = convert_history(conversation_history)
    messages.append(HumanMessage(content=user_prompt))
    tool_events = []

    max_iterations = 8

    for _ in range(max_iterations):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return {
                "response": response.content or "",
                "tool_events": tool_events,
                "available_tools": server_tool_names,
            }

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args") or {}
            tool_call_id = tool_call["id"]
            server_name = tool_to_server.get(tool_name, "unknown")

            event = {
                "name": tool_name,
                "server": server_name,
                "arguments": tool_args,
                "result": None,
                "error": None,
                "status": "Running",
            }
            tool_events.append(event)

            try:
                selected_tool = named_tools.get(tool_name)
                if selected_tool is None:
                    raise ValueError(f"Unknown tool requested: {tool_name}")

                result = await selected_tool.ainvoke(tool_args)
                event["result"] = result
                event["status"] = "Completed"

                if isinstance(result, str):
                    tool_content = result
                else:
                    tool_content = json.dumps(
                        result, ensure_ascii=False, default=str
                    )

            except Exception as exc:
                event["error"] = str(exc)
                event["status"] = "Failed"
                tool_content = json.dumps({"error": str(exc)})

            messages.append(
                ToolMessage(
                    content=tool_content,
                    tool_call_id=tool_call_id,
                )
            )

    return {
        "response": (
            "Maximum tool-call iterations reached. "
            "Please try a more specific request."
        ),
        "tool_events": tool_events,
        "available_tools": server_tool_names,
    }


# ==========================================================
# 6. DISPLAY HELPERS
# ==========================================================

SERVER_LABELS = {
    "math": "🧮 Math MCP: Local",
    "weather": "🌦️ Weather MCP: Online",
}


def display_tool_events(tool_events):
    for event in tool_events:
        server_name = event.get("server", "unknown")
        server_label = SERVER_LABELS.get(
            server_name, f"🔌 {server_name} MCP"
        )

        with st.expander(
            f"🔧 {server_label} → {event['name']}",
            expanded=True,
        ):
            st.markdown(f"**MCP Server:** {server_label}")
            st.markdown(f"**Tool Name:** `{event['name']}`")
            st.markdown(f"**Status:** {event.get('status', 'Unknown')}")

            st.markdown("### Tool Arguments")
            st.json(event.get("arguments", {}))

            if event.get("error"):
                st.error(event["error"])
                continue

            st.markdown("### Tool Result")
            result = event.get("result")

            if isinstance(result, str):
                st.code(result, language="text")
            elif result is not None:
                st.json(result, expanded=False)


# ==========================================================
# 7. STREAMLIT UI
# ==========================================================

st.set_page_config(
    page_title="MCP Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 MCP Chatbot")
st.caption("OpenRouter + Local Math MCP + Online Weather MCP")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "server_tool_names" not in st.session_state:
    st.session_state.server_tool_names = {}
if "mcp_ready" not in st.session_state:
    st.session_state.mcp_ready = False
if "mcp_error" not in st.session_state:
    st.session_state.mcp_error = None


with st.sidebar:
    st.header("⚙️ MCP Configuration")
    st.write("🧮 Math MCP: Local / stdio")
    st.write("🌦️ Weather MCP: Online / Streamable HTTP")
    st.write("🌦️ TO-DO MCP: Online(Own Server) / Streamable HTTP")
    st.write("🧠 LLM: OpenRouter")
    st.caption(
        f"Model: {os.getenv('OPENROUTER_MODEL', 'openai/gpt-oss-120b')}"
    )
    st.divider()

    # This button only tests discovery and stores plain tool names.
    # It does not store MCP clients or bound tools across event loops.
    if st.button("🔌 Test MCP Connections", use_container_width=True):
        try:
            with st.spinner("Discovering MCP tools..."):
                _, _, _, discovered_names = asyncio.run(discover_mcp_tools())
            st.session_state.server_tool_names = discovered_names
            st.session_state.mcp_ready = True
            st.session_state.mcp_error = None
            st.success("MCP tool discovery succeeded.")
        except Exception as exc:
            st.session_state.mcp_ready = False
            st.session_state.mcp_error = str(exc)
            st.error("MCP connection/discovery failed.")

    if st.session_state.mcp_ready:
        st.success("🟢 MCP tools discovered")
        st.markdown("### 🧮 Math MCP: Local")
        math_tools = st.session_state.server_tool_names.get("math", [])
        if math_tools:
            for name in math_tools:
                st.write(f"🔧 `{name}`")
        else:
            st.caption("No Math tools discovered.")

        st.divider()
        st.markdown("### 🌦️ Weather MCP: Online")
        weather_tools = st.session_state.server_tool_names.get("weather", [])
        if weather_tools:
            for name in weather_tools:
                st.write(f"🔧 `{name}`")
        else:
            st.caption("No Weather tools discovered.")

    if st.session_state.mcp_error:
        st.error(st.session_state.mcp_error)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            display_tool_events(message.get("tool_events", []))


user_prompt = st.chat_input(
    "Ask about weather or perform a calculation..."
)

if user_prompt:
    with st.chat_message("user"):
        st.markdown(user_prompt)

    previous_history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages
    ]
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("assistant"):
        with st.spinner("Connecting to MCP tools and asking OpenRouter..."):
            try:
                # Everything async for this turn happens in this one event loop.
                result = asyncio.run(
                    run_chat_turn(
                        user_prompt=user_prompt,
                        conversation_history=previous_history,
                    )
                )

                response_text = result["response"]
                tool_events = result["tool_events"]
                st.session_state.server_tool_names = result["available_tools"]
                st.session_state.mcp_ready = True
                st.session_state.mcp_error = None

                st.markdown(response_text)
                display_tool_events(tool_events)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "tool_events": tool_events,
                })

            except Exception as exc:
                error_message = f"Error processing request: {exc}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "tool_events": [],
                })
