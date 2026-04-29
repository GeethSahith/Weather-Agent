from typing import Annotated, Sequence,TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import requests
import os

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
@tool
def get_weather(city: str)->str:
    """Get the weather for a specific city.
    Args:
        city: The name of the city.
    """
    api_key = os.getenv('OPEN_WEATHER_API')
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            return f"Error: Unable to get weather data"
        temp = data["main"]["temp"]
        temp = round(temp - 273, 1)
        desc = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        city_name = data["name"]

        return f"Weather in {city_name}: {temp} C, {desc} and humidity {humidity}%"

    except Exception as e:
        return str(e)
tools = [get_weather]
model = ChatOllama(model="llama3.1").bind_tools(tools)



def agent(state: AgentState) -> AgentState:
    """invokes the LLM with system prompt + message history."""
    system_prompt = SystemMessage(content="""
        You are weather_bot, a helpful assistant that tells the weather of a city.
        1. Call the Weather tool when only required and respond as a normal llm without calling weather tool when not required. No need to call it for general questions.
        2. If the user wants to get the weather for a specific city, use the 'get_weather' tool with the name of the city.
        3. When the get_weather tool is used, present the weather information clearly. Keep the response short, relevant to the weather, and easy to follow.
        """)
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"\n[Tool Response]: {msg.content}\n")
    messages = [system_prompt] + list(state["messages"])
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """check if the last AI message has tool calls."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()

user_input = input("You: ")
result = app.invoke({"messages": [HumanMessage(content=user_input)]})
print(result["messages"][-1].content)
