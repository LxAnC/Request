import requests


url = 'https://localhost:5000/lm'

data ={
    "messages": [
        {
            "role": "user",
            "content": "你好，怎xxxxx"
        }
    ],
    "model": "gpt-3.5-turbo"
}



ab = requests.post(url,json=data)


msg = ab.json()['choices'][0]['message']['content']

print(msg)