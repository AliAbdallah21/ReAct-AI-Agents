# ReAct AI Agents with LangGraph

A Python implementation of ReAct (Reasoning and Acting) AI agents using LangGraph, demonstrating how to build intelligent agents that can think step-by-step and use tools to accomplish complex tasks.

## Overview

This project showcases how to create AI agents that follow the ReAct paradigm - combining reasoning and acting in language models. The agents can search for information, make decisions, and provide recommendations based on real-time data.

## Features

- **ReAct Agent Architecture**: Implements the ReAct pattern where agents reason about problems and use tools
- **Web Search Integration**: Uses Tavily API for real-time web searching
- **Clothing Recommendations**: Provides weather-appropriate clothing suggestions
- **LangGraph Workflow**: Utilizes LangGraph for complex agent orchestration
- **Interactive Streaming**: Real-time conversation flow with tool usage
- **Visual Workflow**: Graphical representation of the agent's decision process

## Architecture

The system consists of three main components:

1. **Agent Node**: Uses GPT-4o-mini to process queries and decide on tool usage
2. **Tool Node**: Executes tools (web search, clothing recommendations)
3. **Conditional Logic**: Determines whether to continue with tools or end conversation

## Prerequisites

- Python 3.8+
- OpenAI API key
- Tavily API key
- Google Colab environment (for the provided notebook)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AliAbdallah21/ReAct-AI-Agents.git
cd ReAct-AI-Agents
```

2. Install required dependencies:
```bash
pip install langgraph==0.3.34 langchain-openai==0.3.14 langchainhub==0.1.21 langchain==0.3.24 pygraphviz==1.14 langchain-community==0.3.23
```

3. Install system dependencies (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install -y graphviz libgraphviz-dev pkg-config
```

## API Keys Setup

You'll need to obtain API keys for:

- **OpenAI API**: Get your key from [OpenAI Platform](https://platform.openai.com/)
- **Tavily API**: Sign up at [Tavily](https://tavily.com/) for web search functionality

In Google Colab, store these as secrets:
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

## Usage

### Basic Example

```python
from langchain_core.messages import HumanMessage

# Initialize the agent
inputs = {
    "messages": [
        HumanMessage(content="What's the weather like in Zurich, and what should I wear?")
    ]
}

# Run the agent
for output in graph.stream(inputs, stream_mode="values"):
    last_message = output["messages"][-1]
    last_message.pretty_print()
```

### Custom Tools

The project includes two main tools:

1. **Web Search Tool**: Searches the internet for current information
2. **Clothing Recommendation Tool**: Provides clothing suggestions based on weather

```python
@tool
def search_tool(query: str):
    """Search the web for information using Tavily API."""
    return search.invoke(query)

@tool
def recommend_clothing(weather: str) -> str:
    """Returns clothing recommendations based on weather description."""
    # Implementation logic here
```

## Workflow Visualization

The agent follows this decision flow:

```
Start → Agent → Tool Needed? → Tools → Agent → End
         ↑                        ↓
         └────────────────────────┘
```

## Example Interactions

**Input**: "What's the weather like in Tokyo today?"

**Agent Process**:
1. Receives query and determines web search is needed
2. Uses search tool to find current weather information
3. Processes results and provides formatted response

**Input**: "What should I wear in rainy weather?"

**Agent Process**:
1. Analyzes the weather condition
2. Uses clothing recommendation tool
3. Provides appropriate clothing suggestions

## Key Components

### Agent State
```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

### Main Functions
- `call_model()`: Invokes the LLM with current conversation state
- `tool_node()`: Executes tool calls from the last message
- `should_continue()`: Determines whether to continue with tools or end

### Graph Construction
```python
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "end": END,
})
```

## Project Structure

```
ReAct-AI-Agents/
├── Build_ReAct_AI_Agents_with_LangGraph.ipynb
├── README.md
└── requirements.txt (recommended)
```

## Technologies Used

- **LangGraph**: For building stateful, multi-actor applications with LLMs
- **LangChain**: For LLM application development
- **OpenAI GPT-4o-mini**: As the reasoning engine
- **Tavily API**: For web search capabilities
- **Python**: Core programming language

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

**Ali Abdallah**
- Email: Aliabdalla2110@gmail.com
- LinkedIn: [Ali Abdallah](https://www.linkedin.com/in/ali-abdallah-b5ba792b6/)
- GitHub: [@AliAbdallah21](https://github.com/AliAbdallah21)

## Acknowledgments

- Built using LangGraph framework
- Inspired by the ReAct paper on reasoning and acting in language models
- Weather data provided by Tavily search API

---

*This project demonstrates the power of combining reasoning and acting in AI agents, showcasing how modern LLMs can be enhanced with tool usage capabilities for real-world applications.*
