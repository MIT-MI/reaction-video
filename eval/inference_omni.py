import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# default: Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("/orcd/scratch/seedfund/001/multimodal/qua/huggingface/hub/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")

# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-Omni-7B",
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

conversation = [
    {
      "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the human speech in the video."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "file:///orcd/home/002/qua/code/reaction/reaction-video/test_videos/class.mp4"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "In the video, a teacher is seen standing in front of a classroom filled with students. The teacher is explaining a concept related to mathematics, specifically focusing on geometry. The students are attentively listening and taking notes as the teacher uses visual aids to illustrate the points being discussed."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the human speech in the video."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "file:///orcd/home/002/qua/code/reaction/reaction-video/test_videos/a.mp4"},
        ],
    },
]

# set use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

import pdb; pdb.set_trace()

# Inference: Generation of the output text and audio
text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)

response = processor.batch_decode(text_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("Response Text:")
print(response)
