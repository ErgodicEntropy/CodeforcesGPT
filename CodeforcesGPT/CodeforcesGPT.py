from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain import OpenAI
from langchain.llms import GPT4All
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, load_tools, Tool
import streamlit as st
from pydantic import BaseModel, Field
from langchain.chains import PALChain, LLMMathChain, LLMChain
from PromptTemplates import Codeforces_Prompt, Self_Evaluate_Prompt, Zero_Shot_Prompt, Tree_Of_Thought_Prompt, Genetic_Prompting
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
os.environ["OPENAI_API_KEY"] = "Insert your OpenAI_SAPI_Key"


llm = ChatOpenAI(model_name= 'text-davinci-002', temperature= 0.9, verbose = True, max_tokens = 300)

local_path = (
    "D:/nomic.ai/GPT4All/ggml-gpt4all-j-v1.3-groovy.bin"  # replace with your desired local file path
)
callbacks = [StreamingStdOutCallbackHandler()]
#llm = GPT4All(model=local_path, callbacks=callbacks, backend="gptj", verbose=True)


class CalculatorInput(BaseModel):
    question: str = Field()
    
llm_math_chain = LLMMathChain(llm=llm, verbose = True)


memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history",return_messages=True)
tool_names =  ["llm-math","pal-math","pal-colored-objects", "python_repl","wikipedia", "terminal"]

tools = load_tools(tool_names, llm=llm)

tools.append( Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput
    ),
             )






# Non-Functional Quality (Accuracy)

Self_Evaluation_Agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, prompt= Self_Evaluate_Prompt, memory=memory)
Zero_Shot_Chain = PALChain.from_math_prompt(llm, prompt =Zero_Shot_Prompt, verbose= True)
Tree_Of_Thought_Chain = PALChain.from_math_prompt(llm, prompt =Tree_Of_Thought_Prompt, verbose= True)

GenecticAlgorithm_Chain = PALChain.from_math_prompt(llm, prompt=Genetic_Prompting, verbose=True)

# Functionality
CodeforcesAgent= initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, prompt= Codeforces_Prompt, memory=memory)
Codeforces_chain = PALChain.from_math_prompt(llm,verbose = True)


st.title("CodeforcesGPT💡")
st.write("Exploit the power of LLMs to solve Codeforces problems !!!")
Ask_For_TDHelp = st.checkbox("Help")

Problem_Statement = st.text_area("Insert your Codeforces problem statement!")

Codeforces_Solution = Codeforces_chain.run(Problem_Statement)
st.write(Codeforces_Solution)

Enough_Checker = st.radio("Is this enough?", ["Yes", "No"])
while Enough_Checker == "Yes":
    Intermediary_Solution1 = Zero_Shot_Chain.run(Problem_Statement)
    Intermediary_Solution2 = Tree_Of_Thought_Chain.run(Problem_Statement)
    Codeforces_Solution1 = Self_Evaluation_Agent.run(Intermediary_Solution1)
    Codeforces_Solution2 = Self_Evaluation_Agent.run(Intermediary_Solution2)
    Codeforces_Solution = GenecticAlgorithm_Chain.predict(Codeforces_Solution1, Codeforces_Solution2)
    st.write(Codeforces_Solution)
    
if Ask_For_TDHelp:
    your_question = st.text_input("Anything I can help with? What is your question?")
    resp = CodeforcesAgent.run(input=your_question)
    st.write(resp)
    