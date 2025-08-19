import boto3
import json

prompt_data="""
Act as a Shakespeare and write a poem on machine learning
"""

bedrock_client = boto3.client(service_name="bedrock-runtime")

payload={
    "prompt" : "[INST]" + prompt_data + "[/INST]",
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

body=json.dumps(payload)
model_id="meta.llama3-8b-instruct-v1:0"  # Using Llama 3 8B which supports on-demand throughput

# Alternative model IDs that support on-demand throughput:
# "meta.llama2-70b-chat-v1"             # Llama 2 70B Chat
# "meta.llama2-13b-chat-v1"             # Llama 2 13B Chat  
# "anthropic.claude-3-sonnet-20240229-v1:0"  # Claude 3 Sonnet
# "anthropic.claude-3-haiku-20240307-v1:0"   # Claude 3 Haiku

response = bedrock_client.invoke_model(
    modelId=model_id,
    body=body,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get("body").read())
response_text = response_body["generation"]
print(response_text)