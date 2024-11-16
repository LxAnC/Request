import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from flask import Flask, request, jsonify

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("./glm-4-9b-chat-1m", trust_remote_code=True, device_map='auto')
model = AutoModelForCausalLM.from_pretrained(
        "./glm-4-9b-chat-1m",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='auto'
    ).to(device).eval()

# 创建flask实例对象
app = Flask(__name__)


@app.route('/')
def h():
    print('nihaoahoahoah')
    return 'Success,wiat'


@app.route('/hello', methods=['GET'])
def hello_world():
    print('sdsd')
    return 'sdsd,sdsds'
# 注释的都是自己写错的，致谢向航哥

@app.route('/lm', methods=['POST'])
def lm():
    data = request.json  # 直接拿到整个的json串
    query = data['messages'][0]['content'] # 抽取其中的信息

    # print("ok")  test
    # query = request.get_json()
    # print(query)
    # query = query['messages'][0]['content']
    # print(query)
    result = model_inference(query)   #调用模型输入数据
    # print(result)
    # result = result.encode().decode('unicode-escape')
    # print(result)
    out = {
        "id": "QA-medical",
        "object": "chat.completion",
        "created": 0,
        "model": "None",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "user",
                    "content": result
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    #   上面是给入post后json格式




    # out['choices'][0]['message']['content'] = result
    # print(out["choices"][0]["message"]["content"])
    return out


def model_inference(input_data):
    query = input_data
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           )
    inputs = inputs.to(device)

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

        return tokenizer.decode(outputs[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



