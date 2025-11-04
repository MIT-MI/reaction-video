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
                        # {"type": "video_url", "video_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"}},
                        # {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/cough.wav"}},
                        # {"type": "video_url", "video_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
                        {"type": "video_url", "video_url": {"url": video_path}},
                        {"type": "text", "text": text}
                ]
            }
        ],
        temperature=0.2,
        # repetition_penalty = 1.0
        # max_tokens=512,
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