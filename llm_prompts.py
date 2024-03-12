from pydantic import BaseModel, Field
import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "API_KEY"

API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI()


class SummaryPrompt(BaseModel):
    content: list = Field(..., description="The content to be summarized")
    level_of_detail: str = Field("concise", description="The level of detail for the summary")

    def transcript_summary(self):
        detail = ""
        if self.level_of_detail == "concise":
            detail = "- The summary should be concise and easy to understand, and should not be longer than 100 words."
        elif self.level_of_detail == "medium":
            detail = "- The summary should be comprehensive and easy to understand, and should not be longer than 250 words."
        elif self.level_of_detail == "detailed":
            detail = "- The summary should be comprehensive and easy to understand, and should not be longer than 500 words."

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant who only performs the provided instruction following any provided constraints.""",
                },
                {
                    "role": "user",
                "content": f"""
    Can you provide a comprehensive summary of the given transcript? Please meet the following constraints:
    - The summary should cover all the key points and main ideas presented in the original text
    - The summary should be something that may follow the phrase "In this video..."
    - The summary should condense the information into a concise and easy-to-understand format
    - Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition.
    - The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information.
    - Ensure you reply with the right content, and not anything to do with the prompt.
    {detail}
    Transcript: {self.content}""",
                },
            ],
        )
        return completion.choices[0].message.content
    
    def video_summary(self):
        detail = ""
        if self.level_of_detail == "concise":
            detail = "- The summary should be concise and easy to understand, and should not be longer than 100 words."
        elif self.level_of_detail == "medium":
            detail = "- The summary should be comprehensive and easy to understand, and should not be longer than 250 words."
        elif self.level_of_detail == "detailed":
            detail = "- The summary should be comprehensive and easy to understand, and should not be longer than 500 words."

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant who only performs the provided instruction following any provided constraints.""",
                },
                {
                    "role": "user",
                "content": f"""
        Can you provide a comprehensive summary of the given visual captions of frames of a video? Please meet the following constraints:
    - The summary should cover all the key visual elements and main ideas presented in the original video
    - The summary should be something that may follow the phrase "In this video..."
    - The summary should condense the information into a concise and easy-to-understand format
    - Please ensure that the summary includes relevant details and visual information while avoiding any unnecessary information or repetition.
    - The length of the summary should be appropriate for the length and complexity of the original captions, providing a clear and accurate overview without omitting any important information.
    - Ensure you reply with the right content, and not anything to do with the prompt.
    - If any content is repeated across the captions, plese ensure that its importance is highlighted in the summary but not repeated too much.
    - Ignore information about filenames, etc.
    {detail}
    Transcript: {self.content}""",
                },
            ],
        )
        return completion.choices[0].message.content
    
    def audiovisual_summary(self):
        detail = ""
        if self.level_of_detail == "concise":
            detail = "- The summary should be concise and easy to understand, and should not be longer than 100 words."
        elif self.level_of_detail == "medium":
            detail = "- The summary should be comprehensive and easy to understand, and should not be longer than 250 words."
        elif self.level_of_detail == "detailed":
            detail = "- The summary should be comprehensive and easy to understand, and should not be longer than 500 words."

        completion = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant who only performs the provided instruction following any provided constraints.""",
                },
                {
                    "role": "user",
                "content": f"""
        Provide a comprehensive summary of the given visual descriptions and audio summaries of a video. Please meet the following constraints:
    - The summary should cover all the key visual elements, main ideas, and audio content presented in the original video
    - The summary should be something that may follow the phrase "In this video..."
    - The summary should condense the information into a concise and easy-to-understand format
    - Please ensure that the summary includes relevant details and visual information while avoiding any unnecessary information or repetition.
    - The length of the summary should be appropriate for the length and complexity of the original captions, providing a clear and accurate overview without omitting any important information.
    - Ensure you reply with the right content, and not anything to do with the prompt.
    - If any content is repeated across the captions, plese ensure that its importance is highlighted in the summary but not repeated too much.
    - Both the visual and transcriptional information is provided in order of how it appears in the video - please ensure that the summary reflects this order.
    - The summary should combine both the visual and audio information in a way that makes sense and is easy to understand.
    {detail}
    Content: {self.content}""",
                },
            ],
        )
        return completion.choices[0].message.content
         

        