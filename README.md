### Document QnA Search

- AWS Bedrock 
- Langchain

1. Prepare Documents : Split into chunks -> Create Embeddings (Amazon Titan) -> VectorStore (FAISS)
2. Ask Question : 
    - Question -> VectorStore - > Similarity Search -> Retrieve Chunks -> Prompt -> LLM -> Answer


## Setting Up environment

1.Run below command to create python environment
 ```bash
conda create -p venv python==3.10 -y  
```

2. Create requirements.txt file with required libraries.

3. Activate environment by running below command.
 ```bash
conda activate venv/ 
```
4. Install all libs in requirements.txt using below command.
 ```bash
pip install -r requirements.txt
```

5. Please configure local AWS using below command
 ```bash
aws configure
```

6. Make sure we have signed into aws sso using below command
 ```bash
aws sso login
```

7. Please make sure from AWS console, AWS Bedrock has access to the model used.

## Executing the project

Run 'llama2.py' / stablediffusion.py file using below command from respective folder.
 ```bash
python run llama2.py 
```