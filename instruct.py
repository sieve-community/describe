from openai import OpenAI
from pydantic import BaseModel, Field
import os
import instructor
from typing import Optional, List

'''
Data models for generating summaries
- Context: The context of the video
- VideoContext: The list of contexts for the video
- Summary: The summary of the video
'''

class Context(BaseModel):
    type: str = Field(description="The type of context")
    content: str = Field(description="The content of the context")
    start_time: float = Field(description="The time in seconds when the context appears in the video")
    end_time: float = Field(description="The time in seconds when the context ends in the video")

class VideoContext(BaseModel):
    context_list: List[Context] = Field(description="List of contexts for the video")

class Summary(BaseModel):
    summary: str

def get_summary(context: VideoContext, conciseness: str = "concise", llm_backend: str = "openai", additional_instructions = None) -> Summary:
    if llm_backend == "mixtral":
        API_KEY = os.getenv("TOGETHERAI_API_KEY")
    else:
        API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY or API_KEY == "":
        raise Exception("OPENAI_API_KEY or TOGETHERAI_API_KEY environment variable not set")

    if llm_backend == "mixtral":
        client = OpenAI(api_key=API_KEY, base_url="https://api.together.xyz/v1")
    else:
        client = OpenAI(api_key=API_KEY)

    client = instructor.patch(client)
    
    model = "gpt-4-turbo-preview" if llm_backend == "openai" else "mistralai/Mixtral-8x7B-Instruct-v0.1"

    detail_map = {
        "concise": "The summary should be concise and easy to understand, and should not be longer than 100 words.",
        "medium": "The summary should be easy to understand, and should not be longer than 250 words.",
        "detailed": "The summary should be detailed, comprehensive and easy to understand, and should not be longer than 500 words."
    }

    SYSTEM_PROMPT = f'''
        You write really good Wikipedia and IMDB movie summaries. Provide a comprehensive summary of the given visual descriptions and audio transcripts of a video. Please meet the following constraints:
        - The summary should cover all the key visual elements, main ideas, and audio content presented in the original video
        - The summary should condense the information into a concise and easy-to-understand format
        - Please ensure that the summary includes relevant details and visual information while avoiding any unnecessary information or repetition.
        - The length of the summary should be appropriate for the length and complexity of the original captions, providing a clear and accurate overview without omitting any important information.
        - If any content is repeated across the captions, plese ensure that its importance is highlighted in the summary but not repeated too much.
        - Both the visual and transcriptional information is provided in order of how it appears in the video - please ensure that the summary reflects this order.
        - The summary should combine both the visual and audio information in a way that makes sense and is easy to understand.
        - Ensure you reply with the right content, and not anything to do with these instructions.
        - Ensure you always include some level of detail about the visual content.
        - Do not explicitly mention object positions as coordinates, use them to provide a more spatial description. Do not repeat yourself if the same object is mentioned multiple times.
        - Do not say "This video", "This description", "This scene", or "This summary" in the summary. Just provide the description itself.
        - Do not repeat yourself too much, and be to the point -- while explaining the story well.
        - If there isn't that much detail provided, keep as much of the original information as possible, including the detail of the visual content.
    '''
    detail = detail_map[conciseness]
    
    json_context = context.dict()
    return client.chat.completions.create(
        model=model,
        response_model=Summary,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"{additional_instructions}"
            },
            {
                "role": "user",
                "content": f"{detail}"
            },
            {
                "role": "user",
                "content": f"{json_context}"
            },
        ],
        max_retries=3
    )

class KeyObjects(BaseModel):
    key_objects: list[str]

def get_key_objects(context: VideoContext, llm_backend: str = "openai") -> KeyObjects:
    if llm_backend == "mixtral":
        API_KEY = os.getenv("TOGETHERAI_API_KEY")
    else:
        API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY or API_KEY == "":
        raise Exception("OPENAI_API_KEY or TOGETHERAI_API_KEY environment variable not set")

    if llm_backend == "mixtral":
        client = OpenAI(api_key=API_KEY, base_url="https://api.together.xyz/v1")
    else:
        client = OpenAI(api_key=API_KEY)

    client = instructor.patch(client)
    
    model = "gpt-4-turbo-preview" if llm_backend == "openai" else "mistralai/Mixtral-8x22B-Instruct-v0.1"

    SYSTEM_PROMPT = f'''
    Provide a comma separated list of key objects that might be present in the video based on the context.
    - These are objects that a YOLO model will try to detect
    - Make the object names simple and easy to understand, while still being detailed on things like texture, color, and size
    - Ignore any objects that might not be "physically" present in the video. Do not mention you are ignoring them.
    - Just say the object name without any additional information
    Only return a maximum of 10 key objects that would be important for a video summary and do no reply with anything else.
    '''
    
    json_context = context.dict()
    return client.chat.completions.create(
        model=model,
        # response_model=KeyObjects,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"{json_context}"
            },
        ],
        max_retries=3
    ).choices[0].message.content.strip()
