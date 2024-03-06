import sieve
import os
import numpy as np
import cv2
from openai import OpenAI
import concurrent.futures
import time

pyscenedetect = sieve.function.get("sieve/pyscenedetect")
cogvlm = sieve.function.get("sieve/cogvlm-chat")
whisper = sieve.function.get("sieve/speech_transcriber")


def merge_short_scenes(scenes, min_duration=60):
    # Create a new list to store the merged scenes
    merged_scenes = []
    current_scene = None

    for scene in scenes:
        # Calculate the duration of the current scene
        duration = scene["end_seconds"] - scene["start_seconds"]

        # If the scene is shorter than the minimum duration and there is a current scene to merge with
        if duration < min_duration and current_scene is not None:
            # Merge with the current scene
            current_scene["end_seconds"] = scene["end_seconds"]
            current_scene["end_timecode"] = scene["end_timecode"]
            current_scene["end_frame"] = scene["end_frame"]
        else:
            # If there is a current scene, add it to the merged scenes list before moving on
            if current_scene is not None:
                merged_scenes.append(current_scene)
            # Start a new current scene
            current_scene = scene

    # Add the last scene to the merged scenes list
    if current_scene is not None:
        merged_scenes.append(current_scene)

    # Update scene numbers
    for i, scene in enumerate(merged_scenes, start=1):
        scene["scene_number"] = i

    return merged_scenes

def compute_frames(scenes, level_of_detail="medium"):
    frames = {}
    # Compute just the middle frame for all scenes
    for scene in scenes:
        if level_of_detail == "low":
            # 2 frames per scene, 1/4th and 3/4th of the total no. of frames in the scene
            one_fourth_frame = (
                scene["start_frame"] + (scene["end_frame"] - scene["start_frame"]) // 4
            )
            three_fourths_frame = (
                scene["start_frame"]
                + 3 * (scene["end_frame"] - scene["start_frame"]) // 4
            )
            frames[scene["scene_number"]] = [one_fourth_frame, three_fourths_frame]

        if level_of_detail == "medium":
            # 3 frames per scene, 1/4th, 1/2th and 3/4th of the total no. of frames in the scene
            one_fourth_frame = (
                scene["start_frame"] + (scene["end_frame"] - scene["start_frame"]) // 4
            )
            middle_frame = (
                scene["start_frame"] + (scene["end_frame"] - scene["start_frame"]) // 2
            )
            three_fourths_frame = (
                scene["start_frame"]
                + 3 * (scene["end_frame"] - scene["start_frame"]) // 4
            )
            frames[scene["scene_number"]] = [
                one_fourth_frame,
                middle_frame,
                three_fourths_frame,
            ]

        if level_of_detail == "high":
            # 5 frames per scene, 1/5th, 2/5th, 3/5th, 4/5th and the last frame of the scene
            one_fifth_frame = (
                scene["start_frame"] + (scene["end_frame"] - scene["start_frame"]) // 5
            )
            two_fifths_frame = (
                scene["start_frame"]
                + 2 * (scene["end_frame"] - scene["start_frame"]) // 5
            )
            three_fifths_frame = (
                scene["start_frame"]
                + 3 * (scene["end_frame"] - scene["start_frame"]) // 5
            )
            four_fifths_frame = (
                scene["start_frame"]
                + 4 * (scene["end_frame"] - scene["start_frame"]) // 5
            )
            frames[scene["scene_number"]] = [
                one_fifth_frame,
                two_fifths_frame,
                three_fifths_frame,
                four_fifths_frame,
                scene["end_frame"],
            ]

    return frames


def merge_scene_transcripts(scene_data, transcript_data):
    # Initialize a dictionary to hold the result
    scene_transcripts = {}

    # Iterate through each scene
    for scene in scene_data:
        scene_number = scene["scene_number"]
        scene_start = scene["start_seconds"]
        scene_end = scene["end_seconds"]

        # Filter transcript segments that fall within the current scene's timeframe
        scene_transcript = [
            segment["text"]
            for segment in transcript_data
            if segment["start"] >= scene_start and segment["end"] <= scene_end
        ]

        # Join the filtered segments into a single string and add it to the result dictionary
        scene_transcripts[scene_number] = " ".join(scene_transcript)

    return scene_transcripts


@sieve.function(
    name="visual_summarizer",
    python_packages=["openai", "numpy", "opencv-python"],
    system_packages=["ffmpeg"],
    python_version="3.10",
    environment_variables=[
        sieve.Env(name="OPENAI_API_KEY", description="OpenAI API key")
    ],
)
def main(
    video: sieve.Video,
    min_scene_duration: int = 30,
    summary_length: str = "concise",
    level_of_detail: str = "medium",
):
    import subprocess
    import os

    video_path = video.path
    audio_path = video_path + ".wav"

    subprocess.run(["ffmpeg", "-i", video_path, audio_path, "-y"])

    scenes_output = pyscenedetect.run(sieve.File(path=video_path))
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
    scenes_list = list(scenes_output)

    if len(scenes_list) == 0:  # This checks if scenes_output is empty
        print("No scenes detected. Using default scene data.")
        cap = cv2.VideoCapture(video_path)
        duration_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_timecode = (
            time.strftime("%H:%M:%S", time.gmtime(duration_sec)) + ".000"
        )  # Format end time as HH:MM:SS.000
        scenes_list = [
            {
                "start_seconds": 0.0,
                "end_seconds": duration_sec,
                "start_timecode": "00:00:00.000",
                "end_timecode": end_timecode,
                "scene_number": 1,
                "start_frame": 0,
                "end_frame": total_frames,
            }
        ]
        cap.release()

    scenes_output = merge_short_scenes(scenes_list, min_duration=min_scene_duration)
    while len(scenes_output) > 30:
        print("too many scenes, merging")
        scenes_output = merge_short_scenes(
            scenes_output, min_duration=min_scene_duration + 15
        )

    frames = compute_frames(scenes_output, level_of_detail=level_of_detail)

    scene_transcript = merge_scene_transcripts(scenes_output, transcript)

    scene_analysis = {}
    for scene in frames:
        frame_analysis = {}
        cap = cv2.VideoCapture(video_path)
        for frame_number in frames[scene]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                frame_analysis[frame_number] = cogvlm.push(
                    sieve.Image(array=frame),
                    "What is happening in this image? Write in detail.",
                )
        cap.release()
        scene_analysis[scene] = frame_analysis

    for transcript in scene_transcript:
        scene_analysis[transcript]["transcript"] = scene_transcript[transcript]

    client = OpenAI()
    API_KEY = os.getenv("OPENAI_API_KEY")
    # os.environ["OPENAI_API_KEY"] = "API_KEY"

    scene_summaries = {}
    for scene in scene_analysis:
        for frame in scene_analysis[scene]:
            if type(scene_analysis[scene][frame]) == str:
                print("string test")
            else:
                scene_analysis[scene][frame] = scene_analysis[scene][frame].result()
                print("result test")
            
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant. When provided with a list of captions describing the frames of a certain scene in a video and its transcript, you generate a single detailed summary of that scene by combining the information""",
                },
                {
                    "role": "user",
                    "content": f"""Can you provide a comprehensive summary of the given scene using the provided visual captions and transcript? Please meet the following constraints:
                 - The Summary should cover all the key points and main ideas present in the transcript
                 - The Summary should include sufficient visual elements to provide a clear understanding of the scene
                 - The summary should condense the information into a concise and easy-to-understand format
                 - Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition.
                 - The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information.
                 
                 Captions and transcript: {scene_analysis[scene]}""",
                },
            ],
        )
        scene_summaries[scene] = completion.choices[0].message.content

    print(scene_summaries)

    concise = ""

    if summary_length == "concise":
        concise = "- The Summary should be concise, short, and to the point, avoiding any repetition. "

    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant. When provided with visual and audio summaries of a certain video, you generate a single detailed audiovisual summary of that video by combining the information.""",
            },
            {
                "role": "user",
                "content": f"""Provide a comprehensive summary of the given video using the provided visual and audio summaries of each scene in the video. Please meet the following constraints:
             - The Summary should cover all the key points and main ideas present in the visual and audio summaries
             - The Summary should include sufficient visual elements to provide a clear understanding of the scene
             - The summary should condense the information into a concise and easy-to-understand format
             - Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition.
             - The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information.
             - The summary should blend the visual and transcriptional information together instead of presenting them separately.
             - Keep the tone neutral and informative, avoiding any personal opinions or biases.
             {concise}
             Visual and audio summaries: {scene_summaries}""",
            },
        ],
    )

    video_summary = completion.choices[0].message.content

    return video_summary


if __name__ == "__main__":
    main.run(sieve.Video(path="ltt_test.mp4"), min_scene_duration=30, summary_length="concise", level_of_detail="medium")
