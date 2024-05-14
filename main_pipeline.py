import sieve
from typing import Literal

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

def calculate_keyframes(video_duration, frame_rate, start_time, end_time, visual_detail="high"):
    chunk_duration = end_time - start_time
    quarter_frame = int((start_time + chunk_duration / 4) * frame_rate)
    mid_frame = int((start_time + chunk_duration / 2) * frame_rate)
    three_quarters_frame = int((start_time + 3 * chunk_duration / 4) * frame_rate)

    if video_duration > 5:
        if visual_detail == "ultra":
            frame_numbers = [quarter_frame, mid_frame, three_quarters_frame]
        elif video_duration > 1200:
            frame_numbers = [mid_frame]
        else:
            frame_numbers = [quarter_frame, three_quarters_frame]
    else:
        frame_numbers = [mid_frame]

    return [{'frame_number': fn, 'start_time': start_time, 'end_time': end_time} for fn in frame_numbers]


# Sieve functions
whisper = sieve.function.get("sieve/speech_transcriber")
moondream = sieve.function.get("sieve/moondream")
internlm = sieve.function.get("sieve/internlmx-composer-2q")
cogvlm = sieve.function.get("sieve/cogvlm-chat")
vila = sieve.function.get("sieve/vila")
yoloworld = sieve.function.get("sieve/yolov8")

model_mapping = {"low": moondream, "medium": internlm, "high": cogvlm, "ultra": vila}
file_type = {"low": sieve.File, "medium": sieve.File, "high": sieve.Image, "ultra": sieve.File}

@sieve.function(
    name="describe",
    python_packages=["openai", "numpy", "opencv-python", "decord", "instructor", "scenedetect[opencv]", "nltk"],
    system_packages=["ffmpeg"],
    python_version="3.10",
    run_commands=[
        "python -m nltk.downloader punkt",
    ],
    metadata=metadata,
    environment_variables=[
        sieve.Env(name="OPENAI_API_KEY", description="OpenAI API Key", default=""),
        sieve.Env(name="TOGETHERAI_API_KEY", description="Together API Key", default=""),
    ],
)
def main(
    video: sieve.File,
    conciseness: Literal["concise", "medium", "detailed"] = "concise",
    visual_detail: Literal["low", "medium", "high", "ultra"] = "high",
    spoken_context: bool = True,
    object_context: bool = False,
    detail_boost: bool = False,
    chunk_by_scene: bool = False,
    enable_references: bool = False,
    return_metadata: bool = False,
    image_only: bool = False,
    additional_instructions: str = "",
    llm_backend: Literal["openai", "mixtral"] = "openai",
):
    """
    :param video: The video to be described
    :param conciseness: The level of detail for the final description. Pick from 'concise', 'medium', or 'detailed'.
    :param visual_detail: The level of visual detail for the final description. Pick from 'low', 'medium', 'high', or 'ultra'.
    :param spoken_context: Whether to use the transcript when generating the final description.
    :param object_context: Whether to use object detection when generating the final description. BETA FEATURE.
    :param detail_boost: If true, we prompt the underlying models to return even more details in their responses. This can be useful if the initial responses are too vague or lacking in detail.
    :param enable_references: If true, the function will return the references used to generate the description as citations to each sentence, including the timestamps, visual captions and transcripts. chunk_by_scene is auto-enabled if this option is selected.
    :param image_only: By default, describe makes a combination of calls (some which include OpenAI) that generate the most vivid descriptions. This variable instead allows you to simply sample the middle frame of the video for a pure visual description that is less detailed, but doesn't require any external API calls.
    :param chunk_by_scene: If true, the video will be chunked by scene instead of by 60s intervals. This can be useful for videos with multiple scenes or cuts.
    :param return_metadata: If true, the function will return all the granular data used to generate the description, including the keyframes, visual captions, object detections, and summaries.
    :param additional_instructions: Any additional instructions on the questions to answer or the details to emphasize in the final description.
    :param llm_backend: The backend to use for the LLM model. Pick from 'openai' or 'mixtral'. Requires 3rd party API keys. See the README for more information.
    :return: The description
    """
    detail_prompt = "Caption this scene in vivid detail. Short sentences."

    if visual_detail == "ultra":
        detail_prompt = "Please describe this video in a lot of detail."
    
    if additional_instructions:
        if visual_detail == "ultra":
            detail_prompt = f"Please describe the following: {additional_instructions}"
        else:
            detail_prompt = f"Caption the following about this scene in vivid detail. Short sentences: {additional_instructions}"
    
    if detail_boost:
        if visual_detail == "ultra":
            detail_prompt = "Elaborate on the visual and narrative elements of the video in detail."
        else:
            detail_prompt = f"{detail_prompt}. Be extremely detailed."
    
    if enable_references:
        # for better timestamps we chunk by scene
        chunk_by_scene = True

    # Load the video
    video_path = video.path

    if chunk_by_scene and not image_only:
        from queue import Queue
        from threading import Thread
        scene_detection_result = Queue()
        scene_detection_thread = Thread(target=scene_detection_wrapper, args=(video, scene_detection_result), kwargs={'adaptive_threshold': True})
        scene_detection_thread.start()

    print("Loading video...")
    video_reader, video_duration, fps = get_video_info(video_path)
    chunk_size = 60
    
    print("Calculating keyframes...")
    if not image_only:
        keyframes = []
        scenes = []
        scene_frames = []

        if chunk_by_scene:
            print("Chunking video by scene segments...")
            # Get the result from the Queue
            scene_detection_thread.join()
            scene_future = scene_detection_result.get()
            scenes = list(scene_future)
            
        if len(scenes) == 0:
            if not chunk_by_scene: # default chunking
                for start_time in range(0, int(video_duration), chunk_size):
                    end_time = min(start_time + chunk_size, video_duration)
                    keyframe_calculation = calculate_keyframes(video_duration, fps, start_time, end_time, visual_detail)
                    keyframes.extend(keyframe_calculation)   
                    scene_keyframes = [keyframe['frame_number'] for keyframe in keyframe_calculation]
                    scene_frames.append(scene_keyframes)
            else:  # if no scenes detected (means scene is approx same throughout the video)
                keyframe_calculation = calculate_keyframes(video_duration, fps, 0, video_duration, visual_detail)
                keyframes.extend(keyframe_calculation) 
                scene_keyframes = [keyframe['frame_number'] for keyframe in keyframe_calculation]
                scene_frames.append(scene_keyframes)
                
        else:
            min_scene_duration = max(video_duration / 20, 10)
            # join scenes together
            merged_scenes = []
            current_scene = scenes[0]
            for scene in scenes[1:]:
                if current_scene["end_seconds"] - current_scene["start_seconds"] < min_scene_duration:
                    current_scene["end_seconds"] = scene["end_seconds"]
                else:
                    merged_scenes.append(current_scene)
                    current_scene = scene
            if current_scene["end_seconds"] - current_scene["start_seconds"] >= min_scene_duration:
                merged_scenes.append(current_scene)
            
            scenes = merged_scenes
            for scene in scenes:
                keyframe_calculation = calculate_keyframes(video_duration, fps, scene["start_seconds"], scene["end_seconds"], visual_detail)
                scene_keyframes = [keyframe['frame_number'] for keyframe in keyframe_calculation]
                scene_frames.append(scene_keyframes)
                keyframes.extend(keyframe_calculation)
    else:
        # just get the middle frame
        keyframes = [{ 'frame_number': int(video_duration // 2),
                        'start_time': 0,
                        'end_time': video_duration
        }]
        scene_frames = [[int(video_duration // 2)]]

    def get_relevant_chunk(time):
        if chunk_by_scene and not image_only:
            for index, scene in enumerate(scenes):
                if scene["start_seconds"] <= time <= scene["end_seconds"]:
                    return scene["start_seconds"], scene["end_seconds"], index
        for index, start_time in enumerate(range(0, int(video_duration), chunk_size)):
            end_time = min(start_time + chunk_size, video_duration)
            if start_time <= time <= end_time:
                return start_time, end_time, index
        return 0, video_duration, 0

    if spoken_context and not image_only:
        print("Transcribing the audio...")
        # Transcribe the audio
        transcription_job = whisper.push(video)
    
    import cv2
    import os
    import shutil
    import uuid

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

    # only used if VILA is chosen
    def process_scenes(scene_frames):
        model = model_mapping[visual_detail]
        file_arg = file_type[visual_detail](path=video_path)
        keyframe_numbers = ",".join(map(str, scene_frames))
        return model.push(file_arg, detail_prompt, sampling_strategy=keyframe_numbers)

    from concurrent.futures import ThreadPoolExecutor

    print("Extracting keyframes...")
    keyframe_paths = [extract_keyframe(keyframe['frame_number']) for keyframe in keyframes]

    print("Generating visual captions...")
    with ThreadPoolExecutor() as executor:
        if visual_detail == "ultra":
            captioning_jobs = list(executor.map(process_scenes, scene_frames))
        else:
            captioning_jobs = list(executor.map(process_keyframe, keyframe_paths))
    
    # if image_only, return the captions and keyframes
    if image_only:
        return captioning_jobs[0].result()
    
    from instruct import Context, VideoContext, get_summary, get_key_objects, Summary, get_references
    context_list = []

    if spoken_context and not image_only:
        print("Adding transcript to context...")
        # Transcribe the audio
        for transcript_chunk in transcription_job.result():
            if transcript_chunk["text"] == "":
                continue
            context_list.append(
                Context(
                    id = str(uuid.uuid4())[:8],
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
                id = str(uuid.uuid4())[:8],
                type="visual caption",
                content=caption,
                start_time=keyframe['start_time'],
                end_time=keyframe['end_time']
            )
        )
    
    if object_context:
        print("Adding object detection to context...")
        # get key objects for yolo world to detect
        comma_list = get_key_objects(VideoContext(context_list=context_list), llm_backend=llm_backend)
        print(f"Detecting key objects: {comma_list}")
        models = "yolov8l-world"
        object_detection = yoloworld.push(video, confidence_threshold=0.01, classes=comma_list, models=models, fps = 0.1)
        objects_found = []
        for item in object_detection.result():
            # string representation for frame objects
            frame_number = item["frame_number"]
            boxes = item["boxes"]
            for box in boxes:
                class_name = box["class_name"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                if enable_references:
                    if class_name not in objects_found:
                        context_list.append(
                            Context(
                                id = str(uuid.uuid4())[:8],
                                type="object location",
                                content={
                                    "class": class_name,
                                    "x_coord": center_x,
                                    "y_coord": center_y
                                },
                                start_time=int(round(float(frame_number) / fps)),
                                end_time=int(round(float(frame_number) / fps))
                            )

                        )
                        objects_found.append(class_name)

                else:
                    context_list.append(
                        Context(
                            id = str(uuid.uuid4())[:8],
                            type="object location",
                            content={
                                "class": class_name,
                                "x_coord": center_x,
                                "y_coord": center_y
                            },
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
        return chunk_number, get_summary(VideoContext(context_list=chunk), conciseness, llm_backend=llm_backend, additional_instructions=additional_instructions)

    with ThreadPoolExecutor() as executor:
        summaries = dict(executor.map(generate_summary, split_context.items()))

    summary_context_list = []
    for chunk_number, summary in summaries.items():
        summary_context_list.append(
            Context(
                id = str(uuid.uuid4())[:8],
                type="summary",
                content=summary.summary,
                start_time=get_relevant_chunk(chunk_number)[0],
                end_time=get_relevant_chunk(chunk_number)[1]
            )
        )

    def map_references(summary_list : list):
        # filters the context_list to get only the context_ids that are in the summary references
        # returns a dict of all relevent context
        context_dict = {context.id: context for context in context_list}
        all_context = {}
        for item in summary_list:
            filtered_context_ids = [context_id for context_id in item['context_ids'] if context_id in context_dict]
            item['context_ids'] = filtered_context_ids
            item_context = {context_id: context_dict[context_id].dict() for context_id in filtered_context_ids}
            all_context.update(item_context)
     
        return all_context
    
    if len(summaries) == 1:
        summary =  summary_context_list[0].content
    else:
        print("Generating final summary...")
        summary_context = VideoContext(context_list=summary_context_list)
        summary = get_summary(
            summary_context,
            conciseness,
            llm_backend=llm_backend,
            additional_instructions="These are summaries at various points in the video. Combine them to create a single comprehensive summary in chronological order."
        ).summary
    
    print('Generating references...')
    if enable_references:
        summary_obj = Summary(summary=summary)
        summary_timestamps = get_references(context, summary_obj,llm_backend)
        references = map_references(summary_timestamps.dict()['references'])
        summary_refs = {
            'sentences': summary_timestamps.dict()['references'],
            'references': references
        }

    print("Done!")
    if return_metadata:
        chunk_metadata = []
        for key, context_objs in split_context.items():
            chunk_summary = summaries[key].summary
            chunk_start, chunk_end, chunk_number = get_relevant_chunk(context_objs[0].start_time)
            context_dict = [{"type": obj.type, "content": obj.content, "start_time": obj.start_time, "end_time": obj.end_time} for obj in context_objs]
            chunk_metadata.append({
                "scene_number": chunk_number,
                "start_time": chunk_start,
                "end_time": chunk_end,
                "summary": chunk_summary,
                "context": context_dict,
            })
        if enable_references:
            return tuple([summary] + [summary_refs] + [chunk_metadata])
        else:
            return tuple([summary] + [chunk_metadata])
    else:
        if enable_references:
            return tuple([summary] + [summary_refs])
        else:     
            return summary

if __name__ == "__main__":
    print(main(
        video=sieve.File(url="https://storage.googleapis.com/sieve-prod-us-central1-public-file-upload-bucket/3bb46d4e-0583-4b50-bd2f-64b960f47dab/05cc9bfc-a44c-4c03-94d7-1edff2b7f7c7-input-video.mp4"),
        # conciseness="concise"
    ))
    

    

