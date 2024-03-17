from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./Qwen-1_8B-Chat", trust_remote_code=True)

token_results = tokenizer("有志者，事竟成，破釜沉舟，百二秦关终属楚；苦心人，天不负，卧薪尝胆，三千越甲可吞吴。龘")
print(token_results)