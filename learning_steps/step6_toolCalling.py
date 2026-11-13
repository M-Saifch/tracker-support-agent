import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"

# 1) Mock data + asli Python function
DEVICES = {
    "123": {"battery": 82, "status": "moving"},
    "456": {"battery": 15, "status": "idle"},
}

def get_battery_status(device_id):
    device = DEVICES.get(device_id)
    if device is None:
        return {"error": f"Device {device_id} not found"}
    return {
        "device_id": device_id,
        "battery_percent": device["battery"],
        "status": device["status"],
    }

# 2) Tool ka schema (LLM ko batata hai ke tool kya hai)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_battery_status",
            "description": "Get the current battery percentage and status of a GPS tracker by its device ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "string",
                        "description": "The tracker's device ID, e.g. '123'",
                    }
                },
                "required": ["device_id"],
            },
        },
    }
]

# 3) Naam -> function ka nakhsha
available_functions = {"get_battery_status": get_battery_status}

def run_agent(question):
    messages = [
        {
            "role": "system",
            "content": "You are a support assistant for a GPS tracker company. "
                       "Use tools when you need live device data.",
        },
        {"role": "user", "content": question},
    ]

    # Pehli call: LLM faisla karta hai ke tool chahiye ya nahi
    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=tools, tool_choice="auto"
    )
    msg = response.choices[0].message

    if not msg.tool_calls:
        return msg.content  # tool ki zaroorat nahi thi

    # Tool chalao aur result wapas do
    messages.append(msg)
    for call in msg.tool_calls:
        print(f"[TOOL CALL] {call.function.name}({call.function.arguments})")
        func = available_functions[call.function.name]
        args = json.loads(call.function.arguments)
        result = func(**args)
        print(f"[TOOL RESULT] {result}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            }
        )

    # Dusri call: LLM result padh kar final jawab banata hai
    final = client.chat.completions.create(
        model=MODEL, messages=messages, tools=tools
    )
    return final.choices[0].message.content

print(run_agent("What is the battery level of device 123?"))
print("-----")
print(run_agent("Check battery for device 999"))
print("-----")
print(run_agent("What is the capital of France?"))