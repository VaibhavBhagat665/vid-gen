
import gradio as gr
import os
import sys
from pathlib import Path

# Import our modules
from .model_loader import ModelManager
from .video_generator import VideoGenerator


class VideoGenUI:
    def __init__(self):
        self.model_manager = ModelManager()
        self.current_model = None
        self.current_generator = None
        self.model_name = "zeroscope"
    
    def load_model_if_needed(self, model_name):
        if self.current_model is None or self.model_name != model_name:
            self.current_model = self.model_manager.load_model(model_name)
            self.current_generator = VideoGenerator(self.current_model, model_name)
            self.model_name = model_name
    
    def generate_video_ui(self, prompt, model_name, duration, width, height, fps, steps, seed):
        if not prompt.strip():
            return None, "Please enter a prompt"
        
        try:
            # Load model if needed
            self.load_model_if_needed(model_name)
            
            # Generate filename
            safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')
            output_path = f"outputs/ui_{safe_prompt}_{model_name}.mp4"
            
            # Generate video
            video_path = self.current_generator.generate_video(
                prompt=prompt,
                output_path=output_path,
                duration=duration,
                width=width,
                height=height,
                fps=fps,
                num_inference_steps=steps,
                seed=seed if seed > 0 else None
            )
            
            return video_path, f"Video generated successfully: {video_path}"
            
        except Exception as e:
            return None, f"Error: {str(e)}"


def launch_ui():
    ui = VideoGenUI()
    
    with gr.Blocks(title="LocalVidGen - Text to Video") as interface:
        gr.Markdown("# LocalVidGen - Generate Videos from Text")
        gr.Markdown("Enter a text prompt to generate a short video clip using open-source models.")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="A cat playing with a ball in a sunny garden...",
                    lines=2
                )
                
                model_choice = gr.Dropdown(
                    choices=["zeroscope", "modelscope"],
                    value="zeroscope",
                    label="Model"
                )
                
                with gr.Row():
                    duration = gr.Slider(1, 10, value=3, step=1, label="Duration (seconds)")
                    fps = gr.Slider(4, 16, value=8, step=2, label="FPS")
                
                with gr.Row():
                    width = gr.Slider(256, 768, value=512, step=64, label="Width")
                    height = gr.Slider(256, 768, value=512, step=64, label="Height")
                
                with gr.Row():
                    steps = gr.Slider(10, 50, value=20, step=5, label="Inference Steps")
                    seed = gr.Number(value=-1, label="Seed (-1 for random)")
                
                generate_btn = gr.Button("Generate Video", variant="primary")
            
            with gr.Column():
                video_output = gr.Video(label="Generated Video")
                status_text = gr.Textbox(label="Status", lines=3)
        
        generate_btn.click(
            fn=ui.generate_video_ui,
            inputs=[prompt, model_choice, duration, width, height, fps, steps, seed],
            outputs=[video_output, status_text]
        )
        
        gr.Markdown("""
        ### Tips:
        - First generation will be slow as models download (5-10GB)
        - Lower resolution and steps for faster generation
        - Higher steps generally improve quality
        - GPU recommended for reasonable speed
        """)
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
