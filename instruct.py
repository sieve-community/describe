from openai import OpenAI
from pydantic import BaseModel, Field
import os
import instructor
from typing import Optional, List, Union
import json
from nltk.tokenize import sent_tokenize



'''
Data models for generating summaries
- Context: The context of the video
- VideoContext: The list of contexts for the video
- Summary: The Summary of the video

References models for generating timestamps
- SummaryTimestamps: A List of each sentence with their references
- References: The references used to construct summary sentence.
'''


class Context(BaseModel):
    id: str = Field(description= "The id of the context")
    type: str = Field(description="The type of context")
    content: Union[str,dict] = Field(description="The content of the context")
    start_time: float = Field(description="The time in seconds when the context appears in the video")
    end_time: float = Field(description="The time in seconds when the context ends in the video")

class VideoContext(BaseModel):
    context_list: List[Context] = Field(description="List of contexts for the video")

class Summary(BaseModel):
    summary: str = Field(description="The summary of the video")


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
    
    model = "gpt-4o-2024-05-13" if llm_backend == "openai" else "mistralai/Mixtral-8x7B-Instruct-v0.1"

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
    
    model = "gpt-4o-2024-05-13" if llm_backend == "openai" else "mistralai/Mixtral-8x22B-Instruct-v0.1"

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



class References(BaseModel):
    sentence: str = Field(description="Sentence of the summary for which we need references. ")
    start_time: float = Field(description="The time in seconds when the current sentence appears in the video")
    end_time: float = Field(description="The time in seconds when the current sentence ends in the video")
    context_ids : List[str] = Field(description="The ids of the context used to construct summary sentence")

class SummaryTimestamps(BaseModel):
    references: List[References] = Field(description="List of references for each sentence in the summary")


def get_references(context: VideoContext, summary: Summary, llm_backend: str = "openai") -> SummaryTimestamps:

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
    
    model = "gpt-4o-2024-05-13" if llm_backend == "openai" else "mistralai/Mixtral-8x7B-Instruct-v0.1"

    SYSTEM_PROMPT = f'''
    Provide references for each sentence in the summary given to you. The references should be the context_ids of the contexts used to construct the summary sentence.
    - The references should be in the order of the sentences in the summary
    - Each sentence should have a list of context_ids that were used to construct the sentence
    - A sentence can have multiple references.
    - Do not return same sentence multiple times.
    - Do not return empty sentences.
    - You must provide references for all the sentences in the summary_sentences list, do not miss any sentence.
    - The context_ids should be the ids of the contexts used to construct the summary sentence
    - you must also give start_time and end_time for each sentence these should be inferred from the context start_time and end_time
    - You must provide references for each sentence in the summary do not skip any sentence or provide empty context_ids list
    - Do not include any content given to you in context_list in your response except for the context_ids and the start_time and end_time
    '''
    json_context = context.dict()
    summary_sentences = {"summary_sentences": sent_tokenize(summary.summary)}


    if llm_backend == "openai":
        return client.chat.completions.create(
            model=model,
            response_model=SummaryTimestamps,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"{summary_sentences}"
                },
                {
                    "role": "user",
                    "content": f"{json_context}"
                }

            ],
            max_retries=4
        )
    else:
        response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"{summary_sentences}"
                    },


                    {
                        "role": "user",
                        "content": f"{json_context}"
                    },

                ],
                response_format ={
                "type": "json_object", 
                "schema": SummaryTimestamps.model_json_schema()
                },
                max_retries=4
            )
        summary_obj = SummaryTimestamps.parse_raw(response.choices[0].message.content)
        return summary_obj

    

