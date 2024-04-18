import sieve

metadata = sieve.Metadata(
    title="Generate a Video Description",
    description="Generate a description of a video using both audio and visual context.",
    image=sieve.Image(url="https://storage.googleapis.com/sieve-public-data/describe.jpeg"),
    code_url="https://github.com/sieve-community/describe",
    tags=["Video", "Featured"],
    readme=open("README.md", "r").read(),
)

# Define a wrapper function to call scene_detection and put the result in the Queue
def scene_detection_wrapper(file, result_queue, **kwargs):
    print("Cutting video into scene segments...")
    from scene_detection import scene_detection
    result = list(scene_detection(file, **kwargs))
    result_queue.put(result)
    print("Done cutting video into scene segments")

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
yoloworld = sieve.function.get("sieve/yolov8")

model_mapping = {"low": moondream, "medium": internlm, "high": cogvlm}
file_type = {"low": sieve.File, "medium": sieve.File, "high": sieve.Image}

@sieve.function(
    name="describe",
    python_packages=["openai", "numpy", "opencv-python", "decord", "instructor", "scenedetect[opencv]"],
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
    object_context: bool = False,
    detail_boost: bool = False,
    chunk_by_scene: bool = False,
    return_metadata: bool = False,
    image_only: bool = False,
    additional_instructions: str = "",
    llm_backend: str = "openai",
):
    """
    :param video: The video to be described
    :param conciseness: The level of detail for the final description. Pick from 'concise', 'medium', or 'detailed'
    :param visual_detail: The level of visual detail for the final description. Pick from 'low', 'medium', or 'high'
    :param spoken_context: Whether to use the transcript when generating the final description
    :param object_context: Whether to use object detection when generating the final description. BETA FEATURE.
    :param detail_boost: If true, we prompt the underlying models to return even more details in their responses. This can be useful if the initial responses are too vague or lacking in detail.
    :param image_only: By default, describe makes a combination of calls (some which include OpenAI) that generate the most vivid descriptions. This variable instead allows you to simply sample the middle frame of the video for a pure visual description that is less detailed, but doesn't require any external API calls.
    :param chunk_by_scene: If true, the video will be chunked by scene instead of by 60s intervals. This can be useful for videos with multiple scenes or cuts.
    :param return_metadata: If true, the function will return all the granular data used to generate the description, including the keyframes, visual captions, object detections, and summaries.
    :param additional_instructions: Any additional instructions on the questions to answer or the details to emphasize in the final description
    :param llm_backend: The backend to use for the LLM model. Pick from 'openai' or 'mixtral'. Requires 3rd party API keys. See the README for more information.
    :return: The description
    """

    detail_prompt = "Caption this scene in vivid detail. Short sentences."
    if additional_instructions:
        detail_prompt = f"Caption the following about this scene in vivid detail. Short sentences: {additional_instructions}"
    
    if detail_boost:
        detail_prompt = f"{detail_prompt}. Be extremely detailed."

    # Load the video
    video_path = video.path

    if chunk_by_scene and not image_only:
        from queue import Queue
        from threading import Thread
        scene_detection_result = Queue()
        scene_detection_thread = Thread(target=scene_detection_wrapper, args=(video, scene_detection_result), kwargs={'adaptive_threshold': True})
        scene_detection_thread.start()

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
        scenes = []
        if chunk_by_scene:
            print("Chunking video by scene segments...")
            # Get the result from the Queue
            scene_detection_thread.join()
            scene_future = scene_detection_result.get()
            scenes = list(scene_future)

        if len(scenes) == 0:      
            # use default chunking
            for start_time in range(0, int(video_duration), chunk_size):
                end_time = min(start_time + chunk_size, video_duration)
                keyframes.extend(calculate_keyframes(video_duration, fps, start_time, end_time))        
        else:
            min_scene_duration = max(video_duration / 20, 10)
            # join scenes together
            merged_scenes = []
            current_scene = scenes[0]
            # print(scenes)
            for scene in scenes[1:]:
                if current_scene["end_seconds"] - current_scene["start_seconds"] < min_scene_duration:
                    current_scene["end_seconds"] = scene["end_seconds"]
                else:
                    merged_scenes.append(current_scene)
                    current_scene = scene
            scenes = merged_scenes
            for scene in scenes:
                keyframes.extend(calculate_keyframes(video_duration, fps, scene["start_seconds"], scene["end_seconds"]))
    else:
        # just get the middle frame
        keyframes = [video_duration / 2]
    
    def get_relevant_chunk(time):
        if chunk_by_scene:
            for index, scene in enumerate(scenes):
                if scene["start_seconds"] <= time <= scene["end_seconds"]:
                    return scene["start_seconds"], scene["end_seconds"], index
        for index, start_time in enumerate(range(0, int(video_duration), chunk_size)):
            end_time = min(start_time + chunk_size, video_duration)
            if start_time <= time <= end_time:
                return start_time, end_time, index
        return 0, video_duration, 0
    
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
    
    from instruct import Context, VideoContext, get_summary, get_key_objects
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
    
    if object_context:
        print("Adding object detection to context...")
        # get key objects for yolo world to detect
        comma_list = get_key_objects(VideoContext(context_list=context_list), llm_backend=llm_backend)
        print(f"Detecting key objects: {comma_list}")
        models = "yolov8l-world"
        object_detection = yoloworld.push(video, confidence_threshold=0.01, classes=comma_list, models=models, fps = 0.1)
        for item in object_detection.result():
            # string representation for frame objects
            frame_number = item["frame_number"]
            boxes = item["boxes"]
            for box in boxes:
                class_name = box["class_name"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                context_list.append(
                    Context(
                        type="object location",
                        content=f"class: {class_name}, position: ({center_x}, {center_y})",
                        start_time=int(round(float(frame_number) / fps)),
                        end_time=int(round(float(frame_number) / fps))
                    )
                )

    # sort the context list by time
    context_list = sorted(context_list, key=lambda x: x.start_time)
    context = VideoContext(context_list=context_list)

    print(f"Generating summaries...")
    # generate summaries for every 60 seconds of the video
    # split the context into chunks of 60 seconds
    split_context = {}
    for item in context.context_list:
        chunk_number = get_relevant_chunk(item.start_time)[2]
        if chunk_number not in split_context:
            split_context[chunk_number] = []
        split_context[chunk_number].append(item)
    
    from concurrent.futures import ThreadPoolExecutor

    def generate_summary(chunk_data):
        chunk_number, chunk = chunk_data
        return chunk_number, get_summary(VideoContext(context_list=chunk), conciseness, llm_backend=llm_backend, additional_instructions=additional_instructions).summary

    with ThreadPoolExecutor() as executor:
        summaries = dict(executor.map(generate_summary, split_context.items()))
    
    summary_context_list = []
    for chunk_number, summary in summaries.items():
        summary_context_list.append(
            Context(
                type="summary",
                content=summary,
                start_time=get_relevant_chunk(chunk_number)[0],
                end_time=get_relevant_chunk(chunk_number)[1]
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

    if return_metadata:
        chunk_metadata = []
        for key, context_objs in split_context.items():
            chunk_summary = summaries[key]
            chunk_start, chunk_end, chunk_number = get_relevant_chunk(context_objs[0].start_time)
            context_dict = [{"type": obj.type, "content": obj.content, "start_time": obj.start_time, "end_time": obj.end_time} for obj in context_objs]
            chunk_metadata.append({
                "scene_number": chunk_number,
                "start_time": chunk_start,
                "end_time": chunk_end,
                "summary": chunk_summary,
                "context": context_dict,
            })
        return tuple([summary] + chunk_metadata)
    else:     
        return summary

if __name__ == "__main__":
    # print(main(
    #     video=sieve.File(url="https://storage.googleapis.com/sieve-prod-us-central1-public-file-upload-bucket/a3db863d-aaff-4243-9cd2-846bc9a64925/8888b955-21e9-4e59-b285-bf07437f76b9-input-file.mp4"),
    #     conciseness="concise",
    #     llm_backend="mixtral",
    #     spoken_context=False,
    #     # detail_boost=True,
    #     object_context=True,
    #     chunk_by_scene=True,
    #     return_metadata=True
    # ))

    import os

    dir = "/Users/Mokshith/Desktop/testsdescribe"
    describe = sieve.function.get("sieve-developer/describe")
    for file in os.listdir(dir)[2:6]:
        if file.endswith(".mp4"):
            print(f"Processing {file}")
            print(describe.push(video=sieve.File(path=os.path.join(dir, file)), llm_backend="mixtral"))
