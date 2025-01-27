from openai import OpenAI
import os
import json
import time
import openai
from typing import List
from together import Together
import google.generativeai as genai
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from anthropic import Anthropic

# with open('/afs/crc.nd.edu/user/z/ztan3/Private/PerRecLLM/code/config.json', 'r') as f:
#     config_dict = json.load(f)

# os.environ["OPENAI_API_KEY"] = config_dict['openai_api']
# print(os.environ["OPENAI_API_BASE"])
# client = OpenAI(
#     api_key=os.environ["OPENAI_API_KEY"],
#     base_url=os.environ["OPENAI_API_BASE"],
# )


def openai_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    frequency_penalty: float=0,
    presence_penalty: float=0,
    prompt: str=None,
    n: int=1,
    messages: List[dict]=None,
    stop: List[str]=None,
    json_mode: bool=False,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        frequency_penalty (float, optional): The frequency penalty. Defaults to 0.
        presence_penalty (float, optional): The presence penalty. Defaults to 0.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    # Call openai api to generate aspects
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"system","content":"You are a helpful AI assistant."},
                    {"role":"user","content": prompt}]
    
    if openai.__version__ == "0.28.0":
        response = openai.ChatCompletion.create(
            model=model,
            response_format = {"type": "json_object"} if json_mode else None,
            engine=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            **kwargs,
        )
        contents = []
        for choice in response['choices']:
            # Check if the response is valid
            if choice['finish_reason'] not in ['stop', 'length']:
                raise ValueError(f"OpenAI Finish Reason Error: {choice['finish_reason']}")
            contents.append(choice['message']['content'])
    else:
        nvidia_mode = False 
        # for version > 1.0
        if "deepseek" in model:
            assert os.environ.get("DEEPSEEK_API_KEY") is not None, "Please set DEEPSEEK_API_KEY in the environment variables."
            client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
        elif "yi-" in model:
            assert os.environ.get("YI_API_KEY") is not None, "Please set YI_API_KEY in the environment variables."
            client = OpenAI(api_key=os.environ.get("YI_API_KEY"), base_url="https://api.lingyiwanwu.com/v1")
        elif model.endswith("@nvidia"):             
            assert os.environ.get("NVIDIA_API_KEY") is not None, "Please set NVIDIA_API_KEY in the environment variables."
            client = OpenAI(api_key=os.environ.get("NVIDIA_API_KEY"), base_url="https://integrate.api.nvidia.com/v1")
            model = model.replace("@nvidia", "")
            nvidia_mode = True 
            # print(model, client.api_key, client.base_url)
        else:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ["OPENAI_API_BASE"])
            model = model.split("/")[-1]

        if nvidia_mode:
            # print(f"Requesting chat completion from OpenAI API with model {model}")
            # remove system message
            if messages[0]["role"] == "system":
                messages = messages[1:]
            response = client.chat.completions.create(
                model=model, 
                messages=messages,
                temperature=0.001 if temperature == 0 else temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                # n=n,
                # stop=stop,
                **kwargs,
            )
        elif "gemma" in model:
            if messages[0]["role"] == "system":
                messages[1]['content'] = messages[0]['content'] + "\n\n" + messages[1]['content']
                messages = messages[1:]

            response = client.chat.completions.create(
                model=model, 
                response_format = {"type": "json_object"} if json_mode else None,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                **kwargs,
            )
        else: 
            # print(f"Requesting chat completion from OpenAI API with model {model}")
            response = client.chat.completions.create(
                model=model, 
                response_format = {"type": "json_object"} if json_mode else None,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                **kwargs,
            )
        # print(f"Received response from OpenAI API with model {model}")
        contents = []
        for choice in response.choices:
            # Check if the response is valid
            if choice.finish_reason not in ['stop', 'length']:
                if 'content_filter' in choice.finish_reason:
                    contents.append("Error: content filtered due to OpenAI policy. ")
                else:
                    raise ValueError(f"OpenAI Finish Reason Error: {choice.finish_reason}")
            contents.append(choice.message.content.strip())
    return contents


def mistral_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    prompt: str=None,
    messages: List[dict]=None,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content": prompt}]
    api_key = os.getenv("MISTRAL_API_KEY")
    client = MistralClient(api_key=api_key)
    response = client.chat(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        messages=[ChatMessage(role=message['role'], content=message['content']) for message in messages],
    )

    contents = []
    for choice in response.choices:
        contents.append(choice.message.content)
    return contents

def anthropic_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=512,
    top_p: float=1.0,
    prompt: str=None,
    system_msg: str=None,
    messages: List[dict]=None,
    stop: List[str]=None,
    json_mode: bool=False,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        system_msg (str): The system prompt.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """

    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None and prompt is not None:
        messages = [
            {"role":"user", "content": prompt}
        ] 
    if system_msg is None:
        system_msg = ""
    prefill = "{"
    if json_mode:
        messages.append({"role":"assistant", "content": prefill})
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        max_tokens=max_tokens,
        system=system_msg,
        messages=messages,
        stop_sequences=stop,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )

    contents = [response.content[0].text]
    return contents

def together_chat_request(
    model: str=None,
    engine: str=None,
    temperature: float=0,
    max_tokens: int=4096,
    top_p: float=1.0,
    repetition_penalty: float=0,
    prompt: str=None,
    n: int=1,
    messages: List[dict]=None,
    stop: List[str]=None,
    **kwargs,
) -> List[str]:
    """
    Request the evaluation prompt from the OpenAI API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        engine (str): The engine to use.
        temperature (float, optional): The temperature. Defaults to 0.7.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 800.
        top_p (float, optional): The top p. Defaults to 0.95.
        repetition_penalty (float, optional): The presence penalty. Defaults to 0.
        stop (List[str], optional): The stop. Defaults to None.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    # Call openai api to generate aspects
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"user","content": prompt}]
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    if "gemma-2" in model:
        max_chars = 6000*4
        # num_tokens = len(messages[0]["content"])/4 # estimate the number of tokens by dividing the length of the prompt by 4
        if len(messages[0]["content"]) > max_chars:
            print("Truncating prompt to 6000 tokens")
            messages[0]["content"] = messages[0]["content"][:max_chars] + "... (truncated)"

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
        repetition_penalty=repetition_penalty,
        stop=stop,
        **kwargs
    )
    # print(response.choices[0].message.content)
    contents = []
    for choice in response.choices:
        contents.append(choice.message.content)
    return contents


def google_chat_request(
    model: str=None,
    generation_config: dict=None,
    prompt: str=None,
    messages: List[dict]=None,
) -> List[str]:
    """
    Request the evaluation prompt from the Google API in chat format.
    Args:
        prompt (str): The encoded prompt.
        messages (List[dict]): The messages.
        model (str): The model to use.
        generation_config (dict): Generation configurations.
    Returns:
        List[str]: The list of generated evaluation prompts.
    """
    assert prompt is not None or messages is not None, "Either prompt or messages should be provided."
    if messages is None:
        messages = [{"role":"user","parts": ["You are an AI assistant that helps people find information."]},
                    {"role":"model", "parts": ["Understood."]},
                {"role":"user","parts": [prompt]}]

    api_key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=api_key)
    google_model = genai.GenerativeModel(model)


    response = google_model.generate_content(
        messages,
        generation_config=genai.GenerationConfig(
            max_output_tokens=generation_config['max_output_tokens'],
            temperature=generation_config['temperature'],
            stop_sequences=generation_config['stop_sequences'],
            top_p=generation_config['top_p']
        ),
        request_options={"timeout": 600}
    )
    if len(response.candidates) == 0:
        output = ''
    else:
        candidate = response.candidates[0]
        if candidate.finish_reason != 1 and candidate.finish_reason != 2:
            output = ''
        else:
            output = candidate.content.parts[0].text
    contents = [output]
    return contents




def GetAnswer(message=None,
              prompt=None,
              sys_prompt=None,
              model_id=None, 
              engine='openai', 
              temp=0.6, 
              max_tokens=800,
              n=1
              ):
    # print('------------------------ SYSTEM ---------------------------------')
    # print(message[0]['content'])
    # print('------------------------ CONTENT ---------------------------------')
    # print(message[1]['content'])

    if engine == 'openai':
        output = openai_chat_request(
            model=model_id,
            engine=engine,
            temperature=temp,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
            prompt=prompt,
            n=n,
            messages=message
        )
    elif engine == 'mistral':
        output = mistral_chat_request(
            model=model_id,
            engine=engine,
            temperature=temp,
            max_tokens=max_tokens,
            top_p=1.0,
            prompt=prompt,
            messages=message,
        )
    elif engine=="anthropic":

        output = anthropic_chat_request(
            model=model_id,
            engine=engine,
            temperature=temp,
            max_tokens=max_tokens,
            top_p=1.0,
            prompt=prompt,
            system_msg=sys_prompt,
            messages=message,
        )
    elif engine=='together':
        output = together_chat_request(
            model=model_id,
            engine=engine,
            temperature=temp,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
            prompt=prompt,
            n=n,
            messages=message
        )
    elif engine=="google":
        output = google_chat_request(
            model=model_id,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temp,
                "stop_sequences": None,
                "top_p": 1.0
            },
            prompt=prompt,
            messages=message,
        )
    else:
        print("not supported engine!!!")
    
    print(output)

    return output