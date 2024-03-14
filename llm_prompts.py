from pydantic import BaseModel, Field
import os
from openai import OpenAI

# Ensure the API key is set
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise Exception("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=API_KEY)

class SummaryPrompt(BaseModel):
    content: list = Field(..., description="The content to be summarized")
    level_of_detail: str = Field("concise", description="The level of detail for the summary")

    def _generate_summary(self, model: str, prompt_description: str) -> str:
        detail_map = {
            "concise": "- The summary should be concise and easy to understand, and should not be longer than 100 words.",
            "medium": "- The summary should be easy to understand, and should not be longer than 250 words.",
            "detailed": "- The summary should be comprehensive and easy to understand, and should not be longer than 500 words."
        }
        detail = detail_map.get(self.level_of_detail, "")
        
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who only performs the provided instruction following any provided constraints.",
                },
                {
                    "role": "user",
                    "content": f"{prompt_description}{detail}\nContent: {self.content}"
                },
            ],
        )
        return completion.choices[0].message.content

    def transcript_summary(self) -> str:
        prompt_description = """
        Can you provide a comprehensive summary of the given transcript? Please meet the following constraints:
        - The summary should cover all the key points and main ideas presented in the original text
        - The summary should be something that may follow the phrase "In this video..."
        - The summary should condense the information into a concise and easy-to-understand format
        - Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition.
        - The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information.
        - Ensure you reply with the right content, and not anything to do with the prompt.
        """
        return self._generate_summary("gpt-3.5-turbo", prompt_description)

    def video_summary(self) -> str:
        prompt_description = """
        Can you provide a comprehensive summary of the given visual captions of frames of a video? Please meet the following constraints:
        - The summary should cover all the key visual elements and main ideas presented in the original video
        - The summary should be something that may follow the phrase "In this video..."
        - The summary should condense the information into a concise and easy-to-understand format
        - Please ensure that the summary includes relevant details and visual information while avoiding any unnecessary information or repetition.
        - The length of the summary should be appropriate for the length and complexity of the original captions, providing a clear and accurate overview without omitting any important information.
        - Ensure you reply with the right content, and not anything to do with the prompt.
        - If any content is repeated across the captions, plese ensure that its importance is highlighted in the summary but not repeated too much.
        - Ignore information about filenames, etc."""
        return self._generate_summary("gpt-3.5-turbo", prompt_description)

    def audiovisual_summary(self) -> str:
        prompt_description = """
        Provide a comprehensive summary of the given visual descriptions and audio transcripts of a video. Please meet the following constraints:
        - The summary should cover all the key visual elements, main ideas, and audio content presented in the original video
        - The summary should be something that may follow the phrase "In this video..."
        - The summary should condense the information into a concise and easy-to-understand format
        - Please ensure that the summary includes relevant details and visual information while avoiding any unnecessary information or repetition.
        - The length of the summary should be appropriate for the length and complexity of the original captions, providing a clear and accurate overview without omitting any important information.
        - If any content is repeated across the captions, plese ensure that its importance is highlighted in the summary but not repeated too much.
        - Both the visual and transcriptional information is provided in order of how it appears in the video - please ensure that the summary reflects this order.
        - The summary should combine both the visual and audio information in a way that makes sense and is easy to understand.
        - Ensure you reply with the right content, and not anything to do with these instructions.
        - Ensure you always include some level of detail about the visual content.
        """
        return self._generate_summary("gpt-4-turbo-preview", prompt_description)
