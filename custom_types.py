from pydantic import BaseModel, Field
from typing import Optional
import cv2
from decord import VideoReader

class Video(BaseModel):
    path: str
    transcript: Optional[list] = Field(default_factory=list)

    def compute_duration(self):
        '''
        Calculate the duration of the video in seconds
        '''
        vr = VideoReader(self.path)
        fps = vr.get_avg_fps()
        frame_count = len(vr)
        duration = frame_count / fps
        return duration
    
    def extract_chunk_durations(self, chunk_size: int):
        '''
        Extract the timestamps for each chunk based on the chunk size in seconds
        '''
        duration = self.compute_duration()
        chunks = []
        current_start = 0
        while current_start < duration:
            current_end = min(current_start + chunk_size, duration)
            chunks.append((current_start, current_end))
            current_start = current_end
        return chunks
    
class VideoChunk(BaseModel):
    chunk_number: int
    start_time: float
    end_time: float
    source_video_path: str
    source_transcript: Optional[list]

    def compute_keyframes(self):
        '''
        Extract keyframes from the chunk
        '''
        vr = VideoReader(self.source_video_path)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        source_video_duration = total_frames / fps
        duration = self.end_time - self.start_time

        # If duration > 20 minutes, use the middle frame of the chunk as the keyframe
        if source_video_duration > 1200:  
            frame_numbers = [int((self.start_time + duration / 2) * fps)]
        # If duration < 20 minutes, extract 1st and 3rd quarter frames of the chunk
        else:
            frame_numbers = [
                int((self.start_time + (duration / 4) * i) * fps) for i in [1, 3]
            ]
        
        # store the keyframes as images
        keyframe_paths = []
        for i, frame_number in enumerate(frame_numbers, start=1):
            frame = vr[frame_number].asnumpy()
            # Save the keyframe as an image after doing BGR to RGB conversion
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{self.chunk_number}_{i}.jpg", frame)
            keyframe_paths.append(f"{self.chunk_number}_{i}.jpg")
        
        return keyframe_paths

    def compute_chunk_transcript(self):
        transcript_parts = [
            segment['text'] for segment in self.source_transcript
            if segment['start'] >= self.start_time and segment['end'] <= self.end_time
        ]
        return " ".join(transcript_parts)
