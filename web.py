import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gradio as gr
from typing import List, Optional, Tuple, Dict
import mdtex2html
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

default_system = 'You are a helpful assistant.'

model_dir = "./Qwen-1_8B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]

def clear_session() -> History:
    return '', []

def modify_system_session(system: str) -> str:
    if system is None or len(system) == 0:
        system = default_system
    return system, system, []


def model_chat(query: Optional[str], history: Optional[History], system: str
) -> Tuple[str, History, str]:
    if query is None:
        query = ''
    if history is None:
        history = []
    gen = model.chat_stream(tokenizer, query, history, system)
    history.append((None, None))
    for response in gen:
        history[-1] = (query, response)
        yield '', history, system


with gr.Blocks() as demo:
    gr.Markdown("""<p align="center"><img src="https://modelscope.cn/api/v1/models/qwen/Qwen-VL-Chat/repo?Revision=master&FilePath=assets/logo.jpg&View=true" style="height: 80px"/><p>""")
    gr.Markdown("""<center><font size=8>Qwen-1_8B-Chat Bot👾</center>""")
    gr.Markdown("""<center><font size=4>通义千问-1_8B（Qwen-1_8B） 是阿里云研发的通义千问大模型系列的18亿参数规模的模型。</center>""")

    with gr.Row():
        with gr.Column(scale=3):
            system_input = gr.Textbox(value=default_system, lines=1, label='System')
        with gr.Column(scale=1):
            modify_system = gr.Button("🛠️ 设置system并清除历史对话", scale=2)
        system_state = gr.Textbox(value=default_system, visible=False)
    chatbot = gr.Chatbot(label='Qwen-1_8B-Chat')
    textbox = gr.Textbox(lines=2, label='Input')

    with gr.Row():
        clear_history = gr.Button("🧹 清除历史对话")
        sumbit = gr.Button("🚀 发送")

    sumbit.click(model_chat,
                 inputs=[textbox, chatbot, system_state],
                 outputs=[textbox, chatbot, system_input])
    clear_history.click(fn=clear_session,
                        inputs=[],
                        outputs=[textbox, chatbot])
    modify_system.click(fn=modify_system_session,
                        inputs=[system_input],
                        outputs=[system_state, system_input, chatbot])

demo.queue(api_open=False).launch(height=800, share=False)