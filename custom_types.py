from pydantic import BaseModel, Field
import cv2
import subprocess
import os
import json

class Video(BaseModel):
    path: str
    transcript: list = Field(default_factory=list)

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
        duration = self.end_time - self.start_time
        frame_numbers = [
            int(((self.start_time + (duration/4) * i) * fps))
            for i in range(1, 4)
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

# Example usage:
video = Video(path='ltt_test.mp4', transcript=[{'start': 0.18, 'end': 1.86, 'text': " we're going to the producer cam dan you ready all"}, {'start': 2.52, 'end': 5.26, 'text': " right here's our live merch message from one of"}, {'start': 5.26, 'end': 11.24, 'text': ' our contest winners who picked up the 3d down jacket and won a trip to lmg headquarters how'}])
chunk_durations = video.extract_chunk_durations(chunk_size=60)  # For example, 60 seconds apart

# Now create VideoChunk instances and use them to generate ChunkSummary instances. This part depends on how the chunks are utilized and stored.
