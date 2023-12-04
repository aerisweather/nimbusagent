import os
import sys

# use this to add the nimbusagent package to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import textwrap
from datetime import datetime
from dotenv import load_dotenv
from nimbusagent.agent.streaming import StreamingAgent
from simple_spinner import SimpleSpinner

MAX_LINE_LENGTH = 100
AGENT_NAME = 'Jack'
MODEL_NAME = 'gpt-4-0613'
TEMPERATURE = 0.5
COLORS = {
    "orange": "\033[38;5;208m",
    "blue": "\033[34m",
    "bold": '\033[1m',
    "reset": '\033[0m'
}
CURSORS = {
    "show": '\033[?25h',
    "hide": '\033[?25l',
    "block": '\033[2 q',
    "underline": '\033[4 q',
    "ibeam": '\033[6 q'
}
INIT_PROMPT = f"Hello I am {AGENT_NAME} a helpful assistant. Let's chat, ask me a question."

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# Initialize Spinner
spinner = SimpleSpinner(f"{COLORS['orange']}Querying {AGENT_NAME}{COLORS['reset']}")


def clear_terminal():
    print('\033c', end='')


def wrap_text(text, width=80):
    return "\n".join(textwrap.fill(line, width) for line in text.splitlines())


def initialize_agent():
    date_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S+00:00")
    sys_prompt = (f"You are {AGENT_NAME}, a useful assistant that likes to chat\n"
                  f"with people. Answer users questions and ask follow up questions\n"
                  f"The current date and time is {date_now}.\n\n")
    my_history = [{"role": "assistant", "content": INIT_PROMPT}]
    return StreamingAgent(system_message=sys_prompt, message_history=my_history,
                          openai_api_key=openai_api_key, model_name=MODEL_NAME,
                          temperature=TEMPERATURE)


def interact_with_chatbot(agent, user_input):
    agent_response = ''
    received_response = False
    current_line_length = 0

    for content in agent.ask(user_input):
        if not received_response:
            spinner.stop()
            sys.stdout.write(f"{CURSORS['show']}{COLORS['bold']}{COLORS['blue']}{AGENT_NAME}:{COLORS['reset']} ")
            received_response = True

        if content:
            agent_response += content
            current_line_length += len(content)
            if current_line_length > MAX_LINE_LENGTH:
                sys.stdout.write("\n")
                current_line_length = len(content)
            sys.stdout.write(content)
            sys.stdout.flush()

    sys.stdout.write("\n\n")
    sys.stdout.flush()
    return agent_response


def main():
    agent = initialize_agent()
    clear_terminal()
    print(f"{COLORS['bold']}{COLORS['blue']}Nimbus:{COLORS['reset']} {INIT_PROMPT}\n")

    while True:
        user_input = input(f"{CURSORS['block']}{COLORS['bold']}{COLORS['orange']}You:{COLORS['reset']} ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        spinner.update("")
        spinner.start()
        interact_with_chatbot(agent, user_input)


if __name__ == '__main__':
    main()
