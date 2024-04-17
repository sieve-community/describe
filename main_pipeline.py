import concurrent.futures
import sieve
import time

metadata = sieve.Metadata(
    title="Generate a Video Description",
    description="Generate a description of a video using both audio and visual context.",
    image=sieve.Image(url="https://storage.googleapis.com/sieve-public-data/describe.jpeg"),
    code_url="https://github.com/sieve-community/describe",
    tags=["Video", "Featured"],
    readme=open("README.md", "r").read(),
)

# Sieve functions
whisper = sieve.function.get("sieve/speech_transcriber")
moondream = sieve.function.get("sieve/moondream")
internlm = sieve.function.get("sieve/internlmx-composer-2q")
cogvlm = sieve.function.get("sieve/cogvlm-chat")

model_mapping = {"low": moondream, "medium": internlm, "high": cogvlm}
file_type = {"low": sieve.File, "medium": sieve.File, "high": sieve.Image}

@sieve.function(
    name="describe",
    python_packages=["openai", "numpy", "opencv-python", "decord"],
    system_packages=["ffmpeg"],
    python_version="3.10",
    metadata=metadata,
    environment_variables=[
        sieve.Env(name="OPENAI_API_KEY", description="OpenAI API Key", default=""),
        sieve.Env(name="TOGETHERAI_API_KEY", description="Together API Key", default=""),
    ],
)
def main(
    video: sieve.File,
    conciseness: str = "concise",
    visual_detail: str = "high",
    spoken_context: bool = True,
    image_only: bool = False,
    additional_instructions: str = "",
    llm_backend: str = "openai",
):
    """
    :param video: The video to be described
    :param conciseness: The level of detail for the final description. Pick from 'concise', 'medium', or 'detailed'
    :param visual_detail: The level of visual detail for the final description. Pick from 'low', 'medium', or 'high'
    :param spoken_context: Whether to use the transcript when generating the final description
    :param image_only: By default, describe makes a combination of calls (some which include OpenAI) that generate the most vivid descriptions. This variable instead allows you to simply sample the middle frame of the video for a pure visual description that is less detailed, but doesn't require any external API calls.
    :param additional_instructions: Any additional instructions on the questions to answer or the details to emphasize in the final description
    :param llm_backend: The backend to use for the LLM model. Pick from 'openai' or 'mixtral'. Requires 3rd party API keys. See the README for more information.
    :return: The description
    """

    detail_prompt = "Caption this scene in vivid detail. Short sentences."
    if additional_instructions:
        detail_prompt = f"Caption the following about this scene in vivid detail. Short sentences: {additional_instructions}"

    # Load the video
    start_time = time.time()
    video_path = video.path

    if image_only:
        # just get the middle frame and pass it to designated model
        # get the middle frame
        from decord import VideoReader
        vr = VideoReader(video_path)
        middle_frame = vr[len(vr) // 2].asnumpy()
        import cv2
        # Save the keyframe as an image after doing BGR to RGB conversion
        middle_frame = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite("middle_frame.jpg", middle_frame)
        model = model_mapping[visual_detail]
        file_arg = file_type[visual_detail](path="middle_frame.jpg")
        more_detailed_prompt = "Caption this scene with the most important details. Be as detailed as possible."
        description = model.push(file_arg, more_detailed_prompt)
        return description.result()

    from custom_types import Video, VideoChunk
    from llm_prompts import SummaryPrompt
    
    print("Transcribing audio...")
    transcription_job = whisper.push(video)
    video = Video(path=video_path)
    # the video is split into "chunks", and each chunk is processed in parallel
    chunk_size = 60 # seconds
    chunk_durations = video.extract_chunk_durations(chunk_size) # Extract the timestamps for each chunk

    def create_chunk(chunk_data, source_transcript=None):
        i, (start, end) = chunk_data
        chunk = VideoChunk(
            chunk_number=i,
            start_time=start,
            end_time=end,
            source_video_path=video_path,
            source_transcript=source_transcript,
        )
        return chunk

    def process_chunk(chunk):
        # Keyframes are extracted from the chunk and used to generate a description
        # Given the video's transcript, only the transcript present in the chunk is computed and used
        keyframe_paths = chunk.compute_keyframes()

        # extract the captions for each keyframe
        captions = []
        for keyframe_path in keyframe_paths:
            model = model_mapping[visual_detail]
            file_arg = file_type[visual_detail](path=keyframe_path)
            keyframe_caption = model.push(file_arg, detail_prompt)
            captions.append(keyframe_caption)
        
        captions_futures = list(captions)
        captions_list = [future.result() for future in captions_futures]
        
        # Generate the visual summary
        visual_summary = list(captions) if video.compute_duration() > 1200 else SummaryPrompt(content=captions_list, level_of_detail=conciseness, custom_detail=additional_instructions, llm_backend=llm_backend).video_summary()

        return {chunk.chunk_number: visual_summary}

    print("Understanding the visual content...")
    # Process each chunk in parallel
    chunk_summaries = []
    chunks = [create_chunk((i, duration)) for i, duration in enumerate(chunk_durations, start=1)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        chunk_futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
    
    # Transcribe the audio
    if spoken_context:
        transcript = []
        for transcript_chunk in transcription_job.result():
            if transcript_chunk["text"] == "":
                continue
            transcript.append(transcript_chunk)
        transcript = [segment["segments"] for segment in transcript]
        # Extract only the start, end, and text for each segment, excluding the "words" part
        transcript = [
            {"start": item["start"], "end": item["end"], "text": item["text"]}
            for sublist in transcript
            for item in sublist
        ]
        video = Video(path=video_path, transcript=transcript)
    else:
        video = Video(path=video_path)
        transcript = []

    chunks = [create_chunk((i, duration), transcript) for i, duration in enumerate(chunk_durations, start=1)]
    chunk_summaries = [future.result() for future in chunk_futures]

    # If spoken context is enabled, combine the visual and audio summaries
    if spoken_context:
        new_chunk_summaries = []
        for chunk_summary, chunk in zip(chunk_summaries, chunks):
            keys = chunk_summary.keys()
            first_key = next(iter(keys))
            new_chunk_summaries.append({first_key: (chunk.compute_chunk_transcript(), chunk_summary[first_key])})
        chunk_summaries = new_chunk_summaries
    
    # Sort the list by the chunk number so they're in order
    sorted_data = sorted(chunk_summaries, key=lambda x: next(iter(x)))

    print("Combining results...")
    # Combine the results into a single summary
    summary = SummaryPrompt(content=sorted_data, level_of_detail=conciseness, custom_detail=additional_instructions, llm_backend=llm_backend).audiovisual_summary()

    print(f"Time taken: {time.time() - start_time}")
    return summary

if __name__ == "__main__":
    describe = sieve.function.get("sieve-internal/describe")
    import os
    for file in os.listdir("tests"):
        print(describe.push(video=sieve.File(path=f"tests/{file}", llm_backend="mixtral")))
