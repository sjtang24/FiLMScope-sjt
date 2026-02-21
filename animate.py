from moviepy.editor import VideoFileClip

# Input and output file paths
input_gif = "reconstructions-stairs.gif"
output_mp4 = "reconstructions-stairs.mp4"
fps = 3  # Set your desired FPS here

# Load GIF
clip = VideoFileClip(input_gif)

# Write to MP4 with specified FPS
clip.write_videofile(
    output_mp4,
    fps=fps,
    codec="libx264",      # Standard MP4 codec
    audio=False,          # GIFs usually don't have audio
    preset="medium",      # Compression quality
    threads=4             # Number of CPU threads
)

print(f"Converted {input_gif} to {output_mp4} at {fps} FPS")