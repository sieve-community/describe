from pydantic import BaseModel, Field
from typing import Optional
import cv2
import subprocess
import os
import json

# Define the custom types for Vidoe and its chunks
class Video(BaseModel):
    path: str
    transcript: Optional[list] = Field(default_factory=list)

    def compute_duration(self):
        cap = cv2.VideoCapture(self.path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        return duration
    
    def extract_chunk_durations(self, chunk_size: int):
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
    source_transcript: list

    def compute_keyframes(self):
        cap = cv2.VideoCapture(self.source_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_video_duration = total_frames / fps
        duration = self.end_time - self.start_time

        # Adjust frame_numbers based on source video duration
        if source_video_duration > 1200:  # If duration > 20 minutes, extract middle frame of the chunk
            frame_numbers = [int((self.start_time + duration / 2) * fps)]
        elif source_video_duration > 300:  # If duration > 5 minutes, extract 1st and 3rd quarter frames of the chunk
            frame_numbers = [
                int((self.start_time + (duration / 4) * i) * fps) for i in [1, 3]
            ]
        else:
            # If duration <= 5 minutes, extract 1st, 2nd, and 3rd quarter frames of the chunk
            frame_numbers = [
                int((self.start_time + (duration / 4) * i) * fps) for i in range(1, 4)
            ]

        keyframe_paths = []
        for i, frame_number in enumerate(frame_numbers, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = cap.read()
            if success:
                cv2.imwrite(f"{self.chunk_number}_{i}.jpg", frame)
                keyframe_paths.append(f"{self.chunk_number}_{i}.jpg")
            else:
                print(f"Failed to extract frame {i} for chunk {self.chunk_number}.")
        
        cap.release()
        return keyframe_paths

    def compute_chunk_transcript(self):
        transcript_parts = [
            segment['text'] for segment in self.source_transcript
            if segment['start'] >= self.start_time and segment['end'] <= self.end_time
        ]
        return " ".join(transcript_parts)
