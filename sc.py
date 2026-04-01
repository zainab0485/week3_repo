from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

summarize_prompt = PromptTemplate.from_template(
    "Summarize these tasks briefly in one sentence:\n{tasks}"
)


classify_prompt = PromptTemplate.from_template(
    "Based on this summary, classify the tasks into Work, Study, and Personal:\n{summary}"
)

priority_prompt = PromptTemplate.from_template(
    "Based on this summary:\n{summary}\n\n"
    "And these categories:\n{categories}\n\n"
    "Assign priority (High, Medium, Low) to the tasks."
)


parser = StrOutputParser()


user_tasks = """
Complete the Python assignment
Go for a 30-minute run
Update the project budget spreadsheet
Book a dental appointment
Review LangChain documentation
"""


summary = (summarize_prompt | llm | parser).invoke({"tasks": user_tasks})


categories = (classify_prompt | llm | parser).invoke({"summary": summary})


priority = (priority_prompt | llm | parser).invoke({
    "summary": summary,
    "categories": categories
})


print("\n--- AI Task Planner Output ---")
print(f"\nSummary:\n{summary}\n")
print(f"Categories:\n{categories}\n")
print(f"Priority:\n{priority}\n")