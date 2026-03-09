#pip install groq
#pip install gradio

import os
from groq import Groq
from pprint import pprint
from dotenv import load_dotenv
import gradio as gr
import requests
import json

load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key = os.getenv("GROQ_API_KEY"))

# Weather Function
def get_weather(location):
  weather_api_key = os.getenv("WEATHER_API_KEY")
  url = f"https://api.weatherapi.com/v1/current.json?key={weather_api_key}&q={location}"
  response = requests.get(url)
  data = json.loads(response.text)
  return {
      "location": data["location"]["name"],
      "temperature": data["current"]["temp_c"]
  }

# pprint(get_weather("Hyderabad"))

# Tool Definition
tools = [
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City name like Mumbai, London"
          }},
        "required": ["location"]
      }}}
]

# Main LLM Function
def chat_with_weather_api(user_input):
    llm_messages = [{"role": "user", "content": user_input}]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=llm_messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    if response_message.tool_calls:

        tool_call = response_message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)

        location = arguments["location"]

        weather_data = get_weather(location)

        llm_messages.append(response_message)

        llm_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(weather_data)
        })

        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=llm_messages
        )

        return final_response.choices[0].message.content

    else:
        return response_message.content


# Gradio Interface    
demo = gr.Interface(
    fn = chat_with_weather_api,
    inputs = gr.Textbox(lines = 4,label = "Ask about the weather in any city you want!",
                         placeholder = "e.g., What's the weather like in London?"),
    outputs = gr.Textbox(lines = 6, label = "Weather Information"),
    title = "Weather Chatbot",
    description = "Ask about the current weather in any city and get real-time data and also chat with agent!"
)

demo.launch(debug = True)
