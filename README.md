from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, create_react_agent, AgentExecutor

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def summarize(text):
    return f"Summary: {text[:50]}..."

def count_words(text):
    return f"Word count: {len(text.split())}"

tools = [
    Tool(
        name="Summarizer",
        func=summarize,
        description="Use this to summarize text"
    ),
    Tool(
        name="WordCounter",
        func=count_words,
        description="Use this to count words in text"
    )
]

agent = create_react_agent(llm, tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

text = "This is a simple example text to test the AI agent functionality."

result = agent_executor.invoke({"input": text})

print(result["output"])
