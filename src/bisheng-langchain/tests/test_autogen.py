import os
from bisheng_langchain.autogen_agents import (AutoGenAssistantAgent,
    AutoGenGroupChatManager,
    AutoGenUserProxyAgent,
    AutoGenChat,
    AutoGenCustomAgent)
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


openai_api_key = os.environ.get('OPENAI_API_KEY', '')
openai_proxy = os.environ.get('OPENAI_PROXY', '')


def test_two_agents():
    user_proxy = AutoGenUserProxyAgent("user_proxy",
                                human_input_mode="NEVER",
                                max_consecutive_auto_reply=10,
                                code_execution_flag=True,
                               )

    assistant = AutoGenAssistantAgent("assistant",
                                model_name='gpt-4',
                                openai_api_key=openai_api_key,
                                openai_proxy=openai_proxy,
                                temperature=0)

    chat = AutoGenChat(user_proxy_agent=user_proxy, recipient=assistant)
    # response = chat.run("Plot a chart of NVDA and TESLA stock price change YTD.")
    response = chat.run("今天的日期是什么，距离春节还有多少天？")
    print(response)


def test_group_agents():
    user_proxy = AutoGenUserProxyAgent("Admin",
                                code_execution_flag=False,
                                system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
                               )

    engineer = AutoGenAssistantAgent("Engineer",
                                model_name='gpt-4',
                                openai_api_key=openai_api_key,
                                openai_proxy=openai_proxy,
                                temperature=0,
                                system_message='''Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
''')

    scientist = AutoGenAssistantAgent("Scientist",
                                model_name='gpt-4',
                                openai_api_key=openai_api_key,
                                openai_proxy=openai_proxy,
                                temperature=0,
                                system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.""")

    planner = AutoGenAssistantAgent("Planner",
                                model_name='gpt-4',
                                openai_api_key=openai_api_key,
                                openai_proxy=openai_proxy,
                                temperature=0,
                                system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.""")

    executor = AutoGenUserProxyAgent("Executor",
                                human_input_mode="NEVER",
                                code_execution_flag=True,
                                system_message="Executor. Execute the code written by the engineer and report the result.")

    critic = AutoGenAssistantAgent("Critic",
                                model_name='gpt-4',
                                openai_api_key=openai_api_key,
                                openai_proxy=openai_proxy,
                                temperature=0,
                                system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.")

    manager = AutoGenGroupChatManager(agents=[user_proxy, engineer, scientist, planner, executor, critic], messages=[], max_round=50,
                                model_name='gpt-4',
                                openai_api_key=openai_api_key,
                                openai_proxy=openai_proxy,
                                temperature=0)

    chat = AutoGenChat(user_proxy_agent=user_proxy, recipient=manager)
    response = chat.run("find papers on LLM applications from arxiv in the last week, create a markdown table of different domains.")
    print(response)


def test_custom_agent():
    system_template = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" in the end when everything is done.
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
    CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
    llm = ChatOpenAI(model="gpt-4-0613", temperature=0.0)

    llm_chain = LLMChain(llm=llm, prompt=CHAT_PROMPT)
    # print(llm_chain.run("今天的日期是什么，距离春节还有多少天？"))

    user_proxy = AutoGenUserProxyAgent("user_proxy",
                            human_input_mode="NEVER",
                            max_consecutive_auto_reply=10,
                            code_execution_flag=True,
                           )

    assistant = AutoGenCustomAgent(
                            "assistant",
                            'A custom agent.',
                            llm_chain.run)
    chat = AutoGenChat(user_proxy_agent=user_proxy, recipient=assistant)
    # response = chat.run("Plot a chart of NVDA and TESLA stock price change YTD.")
    response = chat.run("今天的日期是什么，距离春节还有多少天？")
    print(response)


# test_two_agents()
# test_group_agents()
test_custom_agent()

