from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Tool 1: Summarize
def summarize(text):
    return f"Summary of text: {text[:50]}..."

# Tool 2: Count words
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

agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

text = "This is a simple example text to test the AI agent functionality."

result = agent.run(text)

print(result)
