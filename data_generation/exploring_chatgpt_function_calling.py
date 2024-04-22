import langchain
import openai
import json

# Environment Variables
import os
from dotenv import load_dotenv


def get_current_weather(location, unit):
    """Get the current weather in a given location"""

    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


def main():
    print("main...")
    import openai

    # Actualiza la versión de la librería OpenAI si es necesario

    function_descriptions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The temperature unit to use. Infer this from the users location.",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location", "unit"],
            },
        }
    ]

    user_query = "What's the weather like in San Francisco?"

    # Se usa la función create de Completion con el engine adecuado
    response = openai.Completion.create(
        engine="text-davinci-003",  # Reemplaza con el motor deseado
        prompt=f"** Your assistant:** I can help you get the weather information. \n** User:** {user_query}",
        max_tokens=1000,  # Ajusta el número máximo de tokens a generar
        n=1,
        stop=None,  # Omite el parámetro stop para una conversación fluida
        temperature=0.7,
        function_descriptions=function_descriptions,
    )

    print(response.choices[0].text.strip())

    ai_response_message = response["choices"][0]["message"]
    print(ai_response_message)

    user_location = eval(ai_response_message["function_call"]["arguments"]).get(
        "location"
    )
    user_unit = eval(ai_response_message["function_call"]["arguments"]).get("unit")

    function_response = get_current_weather(
        location=user_location,
        unit=user_unit,
    )
    print(function_response)

    second_response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "user", "content": user_query},
            ai_response_message,
            {
                "role": "function",
                "name": "get_current_weather",
                "content": function_response,
            },
        ],
    )
    print(second_response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    load_dotenv()
    print("not working...")
    main()
