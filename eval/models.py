from openai import OpenAI
import torch


def generate_response(text: str, model: str, video_path: str = None) -> str:
    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                        {"type": "video_url", "video_url": {"url": video_path}},
                        {"type": "text", "text": text}
                ]
            }
        ],
        temperature=0.01
    )
    return resp.choices[0].message.content
    
    
    
if __name__ == "__main__":
    text = "Describe the video in details."
    video_path = "file:///orcd/home/002/qua/code/reaction/reaction-video/test_videos/class.mp4"
    # model = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = "Qwen/Qwen2.5-Omni-7B"
    response = generate_response(text, model, video_path)
    # torch.cuda.empty_cache()
    print("Response:", response)