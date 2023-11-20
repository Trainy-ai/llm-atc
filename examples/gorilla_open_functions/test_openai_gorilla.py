import json
import openai
import re

GPT_MODEL = "gorilla-openfunctions-v1"
openai.api_key = "EMPTY"
openai.api_base = "http://18.221.156.100:8000/v1"

def extract_function_arguments(response):
    """
    Extracts the keyword arguments of a function call from a program given as a string.

    :param response: The response from gorilla open functions LLM
    :return: A list of dictionaries, each containing key-value pairs of arguments.
    """

    # regex to match the name of the function call
    pattern = r"([^\s(]+)\("
    function_name = re.findall(pattern, response)[0]

# Define a regex pattern to match the function call
    # This pattern matches the function name followed by an opening parenthesis,
    # then captures everything until the matching closing parenthesis
    pattern = rf"{re.escape(function_name)}\((.*?)\)"

    # Find all matches in the program string
    match = re.findall(pattern, response)[0]

    arg_pattern = r"(\w+)\s*=\s*('[^']*'|\"[^\"]*\"|\w+)"
    args = re.findall(arg_pattern, match)

    # Convert the matches to a dictionary
    arg_dict = {key.strip(): value.strip() for key, value in args}
    return {
       "name" : function_name,
       "arguments" : arg_dict,
    }

def get_gorilla_response(prompt="Call me an Uber ride type \"Plus\" in Berkeley at zipcode 94704 in 10 minutes", model="gorilla-openfunctions-v1", functions=[]):
  def get_prompt(user_query, functions=[]):
    if len(functions) == 0:
        return f"USER: <<question>> {user_query}\nASSISTANT: "
    functions_string = json.dumps(functions)
    return f"USER: <<question>> {user_query} <<function>> {functions_string}\nASSISTANT: "
  prompt = get_prompt(prompt, functions=functions)
  try:
    completion = openai.ChatCompletion.create(
      model=model,
      temperature=0.0,
      messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content
  except Exception as e:
    print(e, model, prompt)

functions = [
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
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]

response = get_gorilla_response("What's the weather in Los Angeles in degrees Fahrenheit?", functions=functions)
print(extract_function_arguments(response))