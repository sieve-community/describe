import sieve

metadata = sieve.Metadata(
    title="Generate a Video Description",
    description="Generate a description of a video using both audio and visual context.",
    image=sieve.Image(url="https://storage.googleapis.com/sieve-public-data/describe.jpeg"),
    code_url="https://github.com/sieve-community/describe",
    tags=["Video", "Featured"],
    readme=open("README.md", "r").read(),
)

def get_video_info(video_path):
    '''
    Calculate the info of the video
    '''
    from decord import VideoReader
    vr = VideoReader(video_path)
    fps = vr.get_avg_fps()
    frame_count = len(vr)
    duration = frame_count / fps
    return vr, duration, fps

def calculate_keyframes(video_duration, frame_rate, start_time, end_time):
    chunk_duration = end_time - start_time
    # If duration > 20 minutes, use the middle frame of the chunk as the keyframe
    if video_duration > 1200:  
        frame_numbers = [int((start_time + chunk_duration / 2) * frame_rate)]
    # If duration < 20 minutes, extract 1st and 3rd quarter frames of the chunk
    elif video_duration > 5:
        frame_numbers = [
            int((start_time + (chunk_duration / 4) * i) * frame_rate) for i in [1, 3]
        ]
    else:
        # if duration < 5 seconds, extract just the middle frame
        frame_numbers = [int((start_time + chunk_duration / 2) * frame_rate)]

    return frame_numbers

# Sieve functions
whisper = sieve.function.get("sieve/speech_transcriber")
moondream = sieve.function.get("sieve/moondream")
internlm = sieve.function.get("sieve/internlmx-composer-2q")
cogvlm = sieve.function.get("sieve/cogvlm-chat")

model_mapping = {"low": moondream, "medium": internlm, "high": cogvlm}
file_type = {"low": sieve.File, "medium": sieve.File, "high": sieve.Image}

@sieve.function(
    name="describe",
    python_packages=["openai", "numpy", "opencv-python", "decord", "instructor"],
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
    video_path = video.path

    if spoken_context and not image_only:
        print("Transcribing the audio...")
        # Transcribe the audio
        transcription_job = whisper.push(video)
    
    print("Loading video...")
    video_reader, video_duration, fps = get_video_info(video_path)

    chunk_size = 60

    print("Calculating keyframes...")
    if not image_only:
        keyframes = []
        for start_time in range(0, int(video_duration), chunk_size):
            end_time = min(start_time + chunk_size, video_duration)
            keyframes.extend(calculate_keyframes(video_duration, fps, start_time, end_time))
    else:
        # just get the middle frame
        keyframes = [video_duration / 2]
    
    import cv2
    import os
    import shutil

    keyframes_folder = "keyframes"
    if os.path.exists(keyframes_folder):
        shutil.rmtree(keyframes_folder)
    os.makedirs(keyframes_folder)

    def extract_keyframe(keyframe):
        keyframe_path = os.path.join(keyframes_folder, f"keyframe_{keyframe}.jpg")
        frame = video_reader[keyframe].asnumpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(keyframe_path, frame)
        return keyframe_path

    def process_keyframe(keyframe_path):
        model = model_mapping[visual_detail]
        file_arg = file_type[visual_detail](path=keyframe_path)
        keyframe_job = model.push(file_arg, detail_prompt)
        return keyframe_job

    from concurrent.futures import ThreadPoolExecutor

    print("Extracting keyframes...")
    keyframe_paths = [extract_keyframe(keyframe) for keyframe in keyframes]

    print("Generating visual captions...")
    with ThreadPoolExecutor() as executor:
        captioning_jobs = list(executor.map(process_keyframe, keyframe_paths))
    
    from instruct import Context, VideoContext, get_summary
    context_list = []

    if spoken_context:
        print("Adding transcript to context...")
        # Transcribe the audio
        for transcript_chunk in transcription_job.result():
            if transcript_chunk["text"] == "":
                continue
            context_list.append(
                Context(
                    type="audio transcript",
                    content=transcript_chunk["text"],
                    start_time=round(transcript_chunk["segments"][0]["start"]),
                    end_time=round(transcript_chunk["segments"][-1]["end"])
                )
            )
    
    print("Adding visual captions to context...")
    for job, keyframe in zip(captioning_jobs, keyframes):
        caption = job.result()
        context_list.append(
            Context(
                type="visual caption",
                content=caption,
                start_time=keyframe / fps,
                end_time=keyframe / fps
            )
        )

    # sort the context list by time
    context_list = sorted(context_list, key=lambda x: x.start_time)
    context = VideoContext(context_list=context_list)

    print(f"Generating summaries in {chunk_size} second chunks...")
    # generate summaries for every 60 seconds of the video
    # split the context into chunks of 60 seconds
    split_context = {}
    for item in context.context_list:
        chunk_number = int(item.start_time // (chunk_size))
        if chunk_number not in split_context:
            split_context[chunk_number] = []
        split_context[chunk_number].append(item)
    
    from concurrent.futures import ThreadPoolExecutor

    def generate_summary(chunk_data):
        chunk_number, chunk = chunk_data
        if len(chunk) == 1:
            new_additional_instructions = f"{additional_instructions}. Preserve as much detail as possible from the original context."
            # no need to generate a summary for a single context item that is a visual caption
            if chunk[0].type == "visual caption":
                return chunk_number, chunk[0].content
        else:
            new_additional_instructions = additional_instructions
        return chunk_number, get_summary(VideoContext(context_list=chunk), conciseness, llm_backend=llm_backend, additional_instructions=new_additional_instructions).summary

    with ThreadPoolExecutor() as executor:
        summaries = dict(executor.map(generate_summary, split_context.items()))
    
    summary_context_list = []
    for chunk_number, summary in summaries.items():
        summary_context_list.append(
            Context(
                type="summary",
                content=summary,
                start_time=chunk_number * chunk_size,
                end_time=(chunk_number * chunk_size) + chunk_size
            )
        )
    
    if len(summary_context_list) == 1:
        summary = summary_context_list[0].content
    else:
        print("Generating final summary...")
        summary_context = VideoContext(context_list=summary_context_list)
        summary = get_summary(
            summary_context,
            conciseness,
            llm_backend=llm_backend,
            additional_instructions="These are summaries at various points in the video. Combine them to create a single comprehensive summary in chronological order."
        ).summary
    
    print("Done!")
    return summary

if __name__ == "__main__":
    print(main(
        video=sieve.File(url="https://sieve-prod-us-central1-persistent-bucket.storage.googleapis.com/0a27f1ed-b241-4a1e-8b3c-e8aff3b8379c/c4d0a666-d3a5-4503-952a-1ee8787be2bf/7afcfcb1-2704-414f-8253-0617120b9494/tmpgbxp7us0.mp4?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=abhi-admin%40sieve-grapefruit.iam.gserviceaccount.com%2F20240418%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240418T010356Z&X-Goog-Expires=172800&X-Goog-SignedHeaders=host&x-goog-signature=1ef4e1328ec5597307d3955896cb6aae049e8858e32f4177c431da86da002b701ac750cb96a6963587a4a56e58e9065706a56772b11fb45fef6bef249d5172c3ce41854e74cc21552a01024a492e5e5e56d29312bb3fb88cb5be0e5b4f3bf3fcfe6537fb3e9065c0cf4cc4f2a5e310990fb865d3db2318b50665fa6f0324e061bec537557845eebb5fd2c914851900e1236bee3b3f274f58658172526c438132a3897c3bfd0657e0b0f6c81fe4b01c23b23f5ec43f1473f304a03e7e8635cc5725c27f7d7970d125ab008da34a8e2fe580c9dbfe06740860aaecf0b42ad5ec72046bdc2aba66901584b9c0ae96a02c9930553d665c0a59b2d9b51c11690172da"),
        conciseness="concise",
    ))