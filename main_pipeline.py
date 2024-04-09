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
detail_prompt = "Describe this image with just the most important details. Be concise."
file_type = {"low": sieve.File, "medium": sieve.File, "high": sieve.Image}

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
    image_only: bool = False,
):
    """
    :param video: The video to be described
    :param conciseness: The level of detail for the final description. Pick from 'concise', 'medium', or 'detailed'
    :param visual_detail: The level of visual detail for the final description. Pick from 'low', 'medium', or 'high'
    :param spoken_context: Whether to use the transcript when generating the final description
    :param image_only: By default, describe makes a combination of calls (some which include OpenAI) that generate the most vivid descriptions. This variable instead allows you to simply sample the middle frame of the video for a pure visual description that is less detailed, but doesn't require any external API calls.
    :return: The description
    """

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
        more_detailed_prompt = "Describe this video with the most important details. Be as detailed as possible."
        description = model.push(file_arg, more_detailed_prompt)
        return description.result()

    from custom_types import Video, VideoChunk
    from llm_prompts import SummaryPrompt
    
    # Transcribe the audio
    if spoken_context:
        print("Transcribing audio...")
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
        transcript = []
    
    # the video is split into "chunks", and each chunk is processed in parallel
    chunk_size = 60 # seconds
    chunk_durations = video.extract_chunk_durations(chunk_size) # Extract the timestamps for each chunk

    def process_chunk(chunk_data):
        i, (start, end) = chunk_data
        chunk = VideoChunk(
            chunk_number=i,
            start_time=start,
            end_time=end,
            source_video_path=video_path,
            source_transcript=transcript,
        )

        # Keyframes are extracted from the chunk and used to generate a description
        # Given the video's transcript, only the transcript present in the chunk is computed and used
        keyframe_paths = chunk.compute_keyframes()
        chunk_transcript = chunk.compute_chunk_transcript()

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
        visual_summary = list(captions) if video.compute_duration() > 1200 else SummaryPrompt(content=captions_list, level_of_detail=conciseness).video_summary()

        return {i: (chunk_transcript, visual_summary) if spoken_context else visual_summary}

    print("Understanding the visual content...")

    # Process each chunk in parallel
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

if __name__ == "__main__":
    main(video=sieve.File(url="https://storage.googleapis.com/sieve-prod-us-central1-public-file-upload-bucket/c4d968f5-f25a-412b-9102-5b6ab6dafcb4/ededa101-a156-40a6-b670-4567b9b3c372-AnimateDiff_00057.mp4"))