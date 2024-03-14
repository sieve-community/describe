import concurrent.futures
import sieve
import time

metadata = sieve.Metadata(
    description="Describe a video by utilizing its visuals and spoken content",
    readme=open("README.md", "r").read(),
)

# Sieve functions
whisper = sieve.function.get("sieve/speech_transcriber")
moondream = sieve.function.get("sieve/moondream")
internlm = sieve.function.get("sieve/internlmx-composer-2q")
cogvlm = sieve.function.get("sieve/cogvlm-chat")

@sieve.function(
    name="describe",
    python_packages=["openai", "numpy", "opencv-python", "decord"],
    system_packages=["ffmpeg"],
    python_version="3.10",
    metadata=metadata,
    environment_variables=[
        sieve.Env(name="OPENAI_API_KEY", description="OpenAI API Key")
    ],
)
def main(
    video: sieve.File,
    conciseness: str = "concise",
    visual_detail: str = "medium",
    spoken_context: bool = True,
):
    """
    :param video: The video to be described
    :param conciseness: The level of detail for the final description. Pick from 'concise', 'medium', or 'detailed'
    :param visual_detail: The level of visual detail for the final description. Pick from 'low', 'medium', or 'high'
    :param spoken_context: Whether to use the transcript when generating the final description
    :return: The description
    """
    from custom_types import Video, VideoChunk
    from llm_prompts import SummaryPrompt

    # Load the video
    start_time = time.time()
    video_path = video.path

    # Transcribe the audio
    print("Transcribing audio...")
    if spoken_context:
        transcript = []
        for transcript_chunk in whisper.run(sieve.File(path=video_path)):
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
    
    # Extract chunk durations
    chunk_size = 60
    chunk_durations = video.extract_chunk_durations(chunk_size)

    def process_chunk(chunk_data):
        i, (start, end) = chunk_data
        chunk = VideoChunk(
            chunk_number=i,
            start_time=start,
            end_time=end,
            source_video_path=video_path,
            source_transcript=transcript,
        )
        keyframe_paths = chunk.compute_keyframes()
        chunk_transcript = chunk.compute_chunk_transcript()

        detail_prompt = "Describe this image with just the most important details. Be concise."
        model_mapping = {"low": moondream, "medium": internlm, "high": cogvlm}
        file_type = {"low": sieve.File, "medium": sieve.File, "high": sieve.Image}

        captions = []
        for keyframe_path in keyframe_paths:
            model = model_mapping[visual_detail]
            file_arg = file_type[visual_detail](path=keyframe_path)
            keyframe_caption = model.push(file_arg, detail_prompt)
            captions.append(keyframe_caption)
        
        captions_futures = list(captions)
        captions_list = [future.result() for future in captions_futures]
        
        visual_summary = list(captions) if video.compute_duration() > 1200 else SummaryPrompt(content=captions_list, level_of_detail=conciseness).video_summary()

        return {i: (chunk_transcript, visual_summary) if spoken_context else visual_summary}

    print("Understanding the visual content...")
    
    chunk_summaries = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        chunk_futures = [executor.submit(process_chunk, (i, duration)) for i, duration in enumerate(chunk_durations, start=1)]
    chunk_summaries = [future.result() for future in chunk_futures]
    # Sort the list by the chunk number so they're in order
    sorted_data = sorted(chunk_summaries, key=lambda x: next(iter(x)))

    print("Combining results...")
    # Combine the results into a single summary
    summary = SummaryPrompt(content=sorted_data, level_of_detail=conciseness).audiovisual_summary()

    print(f"Time taken: {time.time() - start_time}")
    return summary
