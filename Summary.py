from transformers import AutoTokenizer, AutoModelForCausalLM
import arvix  # Importing your arxiv.py script
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import requests
from Chatbot import run_chatbot_with_document

# Initialize the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to generate summary with GPT-3.5
def generate_summary(text,language,tokenizer,model):
    template = f"""
###Role:
Your task is to answer the Question by using the Reference and make sure to elaborate about it in Output using the Reference.
Make sure it actually answers the Question and make sure to convey it in {language}.
###Reference:
{text}
###Question:Can you make sure summarize this Reference for me to conduct a literature review in 150 words
###Output:
    """
    inputs = tokenizer(f"{template}", return_tensors='pt').input_ids.to('cuda:0')
    outputs = model.generate(input_ids=inputs, max_new_tokens=2048, do_sample=True, top_p=1.0, top_k = 100, temperature=0.65)
    gen_text = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(template):]
    return gen_text  
def find_closest_papers(user_input, num_matches=1):
    # Generate embedding for user input
    user_embedding = model.encode(user_input)

    # Load existing dataset
    df = pd.read_csv("arxiv_papers_with_embeddings.csv")

    # Ensure embeddings are loaded as arrays
    paper_embeddings = df['embeddings'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',') if isinstance(x, str) else x).tolist()

    # Calculate similarities and get top matches
    similarities = cosine_similarity([user_embedding], paper_embeddings)[0]
    top_indices = np.argsort(similarities)[-num_matches:]

    return df.iloc[top_indices]


def main():
    print("=======================================================================================================================")
    user_topic = input("Enter your topic: ")
    language = input("Enter your preferred language: ")
    arvix.main(user_topic)  # Process and fetch papers based on the topic

    df = pd.read_csv("arxiv_papers_with_embeddings.csv")
    closest_papers = find_closest_papers(user_topic,10)  # Assuming this function is defined elsewhere to use 'df'

    tokenizer = AutoTokenizer.from_pretrained("TomGrc/FusionNet_7Bx2_MoE_14B")
    model = AutoModelForCausalLM.from_pretrained("TomGrc/FusionNet_7Bx2_MoE_14B",load_in_4bit=True,device_map='cuda:0')
    count = 0
    for index, paper in closest_papers.iterrows():
        # Assuming generate_summary is correctly defined to use 'tokenizer' and 'model'
        summary = generate_summary(paper['abstract'], language, tokenizer, model)
        count += 1 
        print(f"\nSummary for paper {count} with the title {paper['title']}:\n{summary}")

    choice = int(input("\nEnter the number of the paper you would like to download: ")) - 1
    selected_paper = closest_papers.iloc[choice]

    download_link = f"https://arxiv.org/pdf/{selected_paper['id']}.pdf"
    response = requests.get(download_link)
    if response.status_code == 200:
        filename = "Downloaded.pdf"
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded '{selected_paper['title']}' to {filename}")
        run_chatbot_with_document(filename)  
    else:
        print("Failed to download the paper.")

if __name__ == "__main__":
    main()