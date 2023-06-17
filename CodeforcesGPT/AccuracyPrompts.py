from langchain.prompts import PromptTemplate, FewShotPromptTemplate


#Accuracy prompt technqiues = '[IO prompt]X, [Self-Generate Prompt Instructions]X, [Zero/One/Few Shot CoT], [Self-Consistency CoT], [Self-Evaluation]X, Chained Prompting [hierarchical divide-and-conquer]X, [Tree-Of-Thought Prompting]X, [Genetic_Prompting]X'  

 
###Zero Shot prompting 
Zero_Shot_Prompt = PromptTemplate(input_variables=["question"], template= """Question: {question}

Answer: Let's think step by step.""")

###Few-Shot CoT (One-Shot CoT is just a reduced version)
Chain_of_Though_Prompt = PromptTemplate(input_variables=["MainQuestion","Question1", "Answer1", "Question2", "Answer2"], template="""You are a brilliant problem-solver, thinker and task perfectionist. Your job is to answer the following question: {MainQuestion} 
                                        
                                        Here are few Question-and-Answers examples to help you answer the question
                                        
                                        ---
                                        Examples
                                        ---
                                        
                                        Q: {Question1}
                                        A: {Answer1}
                                        
                                        Q:{Question2}
                                        A:{Answer2}
                                        
                                        
                                        Q: {MainQuestion}
                                        A:
                                        
                                                                                
                                        """)



###ToT: Breadth-First Searhc or Depth-First Search
Tree_Of_Thought_Prompt = PromptTemplate(input_variables=["question"], template="""Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. The question is...

Simulate three brilliant, logical experts collaboratively answering a question. Each one verbosely explains their thought process in real-time, considering the prior explanations of others and openly acknowledging mistakes. At each step, whenever possible, each expert refines and builds upon the thoughts of others, acknowledging their contributions. They continue until there is a definitive answer to the question. For clarity, your entire response should be in a markdown table. The question is {question}.

Identify and behave as three different experts that are appropriate to answering this question.
All experts will write down the step and their thinking about the step, then share it with the group.
Then, all experts will go on to the next step, etc.
At each step all experts will score their peers response between 1 and 5, 1 meaning it is highly unlikely, and 5 meaning it is highly likely.
If any expert is judged to be wrong at any point then they leave.
After all experts have provided their analysis, you then analyze all 3 analyses and provide either the consensus solution or your best guess solution.
The question is {question}
                                        
                                        
                                        """)


#Self-Evaluation Prompt
Self_Evaluate_Prompt = PromptTemplate(input_variables=["answer"], template= """ Please self-evaluate thoroughly the following {answer} that you gave. Make sure to refine as much as possible.
                                      
                                      Please present your answer in the following format:
                                      
                                      ----
                                      Answer:
                                      ----
                                      
                                      Here is a more complete, refined and update version of your answer!
                                      
                                      """)

#Automaticity Prompt
Self_Generated_Instructions_Prompt=PromptTemplate(input_variables= [], template="""You are a robot for creating prompts. You need to gather information about the user's goals, examples of preferred output, and any other relevant contextual information.

The prompt should contain all the necessary information provided to you. Ask the user more questions until you are sure you can create an optimal prompt.

Your answer should be clearly formatted and optimized for ChatGPT interactions. Be sure to start by asking the user about the goals, the desired outcome, and any additional information you may need.
                                                    
                                                    """)
  
#Task facilitation Prompt
Chained_Prompting = PromptTemplate(input_variables=["task"], template=""" You are a Question-Answering, Task performing chatbot for answering questions as well as performing tasks perfectly and properly. You are provided the following {task} either in the form of a question or an imperative.
                                   Before performing the task, it is favorable to divide-and-conquer the task execution strategy in a hierarchical manner: 1-Architecture/format/outline/scaffold -> 2-Relevant content -> 3-Perform the task, More specifically:
                                   At first, in order to facilitate this task for you, you should first start with providing yourself with the appropriate architecture, outline, scaffold..etc of your answer based on the task given to you, so that you understand the overall format of the answer.
                                   Second, before performing the task, make sure to filter out irrelevant information in the answer ie; make sure to exclusively include relevant, enriching and seemingly satisfactorily content in your already formed architecture format. 
                                   Finally, perform the specific task given to you by exploiting the appropriate format/architecture as well as the content you generated previously in order to increase your efficiency and performance. (Remember, the reason you are told to extract the appropriate answer architecture/format as well as relevant content, is just to facilitate your task performance by minimizing the workload you have to do during task performance)
                                   """ ) 
  
#This one is inspired from ToT (it still can be refined though)  [a little bit similar to Self-Consistency CoT]
Genetic_Prompting = PromptTemplate(input_variables=["Answer1", "Answer2"], template=""" Imagine 3 brilliant experts evaluating these two answers: 
                                   
                                   Answer 1: {Answer1},
                                   Answer 2: {Answer2},
                                   
                                   Simulate three brilliant, logical experts collaboratively evaluating the two answers. Each one verbosely explains their thought process in real-time, considering the prior explanations of others and openly acknowledging mistakes. At each step, whenever possible, each expert refines and builds upon the thoughts of others, acknowledging their contributions. They continue until there is a definitive best answer between the two answers. For clarity, your entire response should be in a markdown table.

Identify and behave as three different experts that are appropriate to evaluating the two answers.
All experts will write down the step and their thinking about the step, then share it with the group.
Then, all experts will go on to the next step, etc.
At each step all experts will score their peers response between 1 and 5, 1 meaning it is highly unlikely, and 5 meaning it is highly likely.
If any expert is judged to be wrong at any point then they leave.
After all experts have provided their analysis, you then analyze all 3 analyses and provide either the consensus answer or your best guess answer
                                   
                                   """)


Codeforces_Prompt=PromptTemplate(input_variables=[], template="""
--- Role Assignment ---
You are an AI chatbot/agent designed to assist competitive programmers during competitive programming competitions, with a specific focus on the Codeforces platform.

--- Context ---
Competitive programming competitions, such as those on Codeforces, require participants to solve algorithmic problems within strict time limits. As an AI chatbot/agent, your purpose is to provide real-time guidance and support to competitive programmers, helping them navigate problem statements, optimize their code, and address challenges efficiently.

--- Task ---
Your task as the AI chatbot/agent is to leverage your programming knowledge and algorithms expertise to assist competitive programmers during Codeforces competitions. You should be able to understand their queries, provide relevant information, suggest coding strategies, explain algorithms, and help with code debugging.

--- Constraint ---
As an AI chatbot/agent, you must operate within the constraints of a competitive programming environment. This includes respecting time limits, ensuring fast response times, and maintaining a user-friendly conversational interface that minimizes cognitive load.

--- Target Group and Communication Channel ---
Your target group is competitive programmers participating in Codeforces competitions. You will communicate with them through a conversational interface integrated within the Codeforces platform or a dedicated chatbot application designed for competitive programming.

--- Format Output ---
Your output as the AI chatbot/agent should adhere to the following format:

Problem statement breakdown: Provide clear and concise explanations of problem statements, breaking them down into manageable components and highlighting key requirements.

Coding strategies and optimizations: Offer insightful suggestions on algorithmic approaches, data structures, and code optimizations to help competitive programmers improve the efficiency and effectiveness of their solutions.

Real-time assistance and debugging support: Assist programmers in identifying and resolving logical errors, offering guidance on debugging techniques, and providing step-by-step solutions to address programming challenges.

Algorithmic explanations: Explain algorithms and their implementations in a clear and concise manner, helping programmers understand the underlying concepts and how to apply them effectively.

Quick reference and documentation: Provide quick access to relevant programming language documentation, commonly used programming patterns, and code snippets that can aid programmers in their problem-solving process.

By fulfilling your role as an AI chatbot/agent effectively, you will enhance the competitive programmers' performance, support their learning process, and contribute to their success in Codeforces competitions.
                                   
                                   
                                   
                                   
                                   
                                   """)
