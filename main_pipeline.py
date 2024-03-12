import concurrent.futures
import sieve
import subprocess
from custom_types import Video, VideoChunk
from llm_prompts import SummaryPrompt
import time

metadata = sieve.Metadata(
    description="Summarize a video by generating a transcript and visual summary",
    image=sieve.Image(path="logo.webp"),
    readme=open("README.md", "r").read(),
)

# Sieve functions
whisper = sieve.function.get("sieve/speech_transcriber")
moondream = sieve.function.get("sieve/moondream")
internlm = sieve.function.get("sieve-internal/internlmx-composer-2q")
cogvlm = sieve.function.get("sieve/cogvlm-chat")

@sieve.function(
    name="describe",
    python_packages=["openai", "numpy", "opencv-python"],
    system_packages=["ffmpeg"],
    python_version="3.10",
    metadata=metadata,
)
def main(
    video: sieve.Video,
    summary_type: str = "medium",
    visual_detail: str = "medium",
    include_transcript: bool = True,
):
    """
    :param video: The video to be summarized
    :param summary_type: {concise, medium, detailed} The level of detail for the final summary
    :param visual_detail: {low, medium, high} The level of visual detail for the final summary
    :param include_transcript: Whether to use the transcript when generating the final summary
    :return: The summarized content
    """
    # Load the video
    start_time = time.time()
    video_path = video.path

    # Extract audio
    audio_path = "audio.wav"
    subprocess.run(["ffmpeg", "-i", video_path, audio_path, "-y"])
    
    # Transcribe the audio
    print("Transcribing audio...")
    if include_transcript:
        transcript = []
        for transcript_chunk in whisper.run(sieve.File(path=audio_path)):
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

    # Prepare for parallel execution
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

        if include_transcript:
            transcript_summary = SummaryPrompt(content=list(chunk_transcript), level_of_detail=summary_type).transcript_summary()
        
        chunk_captions = {}

        # Generate captions for each keyframe
        captions = {}
        for keyframe_path in keyframe_paths:
            if visual_detail == "low":
                keyframe_caption = moondream.run(sieve.File(path=keyframe_path), "Describe this image in detail")
            if visual_detail == "medium":
                keyframe_caption = internlm.run(sieve.File(path=keyframe_path), "Describe this image in detail")
            if visual_detail == "high":
                keyframe_caption = cogvlm.run(sieve.Image(path=keyframe_path), "Describe this image in detail")
            captions[keyframe_path] = keyframe_caption

        captions_list = list(captions.values())

        if video.compute_duration() > 1200:
            visual_summary = captions_list
        else:
            visual_summary = SummaryPrompt(content=captions_list, level_of_detail=summary_type).video_summary()

        if include_transcript:
            chunk_captions[i] = transcript_summary, visual_summary
        else:
            chunk_captions[i] = visual_summary

        return chunk_captions

    chunk_summaries = []

    print("Processing chunks...")

    # Process each chunk in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, (i, duration)) for i, duration in enumerate(chunk_durations, start=1)]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                # Process the result if needed
                chunk_summaries.append(result)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    # Sort the list by the chunk number so they're in order
    sorted_data = sorted(chunk_summaries, key=lambda x: next(iter(x)))

    print("Combining results...")

    # Combine the results into a single summary
    summary = SummaryPrompt(content=sorted_data, level_of_detail=summary_type).audiovisual_summary()

    print(f"Time taken: {time.time() - start_time}")
    return summary

if __name__ == "__main__":
    main.run(sieve.Video(path="ltt_test.mp4"), summary_type="medium", visual_detail="medium", include_transcript=True)
