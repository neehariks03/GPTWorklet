import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import requests
from chatbot import run_chatbot_with_document
openai.api_key = "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"
# Initialize the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to generate summary with GPT-4
def generate_summary(text, language):
    try: 
        response = openai.Completion.create(
            engine="Genera",  # Assuming "text-davinci-003" as an example, replace with your actual GPT-4 model name
            prompt=f"Please summarize the following text in {language} within 150 words:\n\n{text}",
            max_tokens=150,
            temperature=0.7,
            top_p=1.0,
            n=1,
            stop=None
        )
        summary = response.choices[0].text.strip()
        return summary
        print(response.choices[0].text.strip())
    except openai.error.InvalidRequestError as e:
        print(f"Invalid request error: {e}")
    except openai.error.AuthenticationError as e:
        print(f"Authentication error: {e}")
    except openai.error.APIConnectionError as e:
        print(f"API connection error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        

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

    # Assuming there's a function to fetch and process papers based on the topic
    # e.g., arxiv.main(user_topic) if you have a module named arxiv doing this

    df = pd.read_csv("arxiv_papers_with_embeddings.csv")
    closest_papers = find_closest_papers(user_topic, 10)

    count = 0
    for index, paper in closest_papers.iterrows():
        summary = generate_summary(paper['abstract'], language)
        count += 1
        print(f"\nSummary for paper {count} with the title {paper['title']}:\n{summary}")

    choice = int(input("\nEnter the number of the paper you would like to download: ")) - 1
    selected_paper = closest_papers.iloc[choice]

    download_link = f"https://arxiv.org/pdf/{selected_paper['id']}.pdf"
    response = requests.get(download_link)
    if response.status_code == 200:
        filename = f"{selected_paper['title'].replace('/', '_')}.pdf"  # Sanitize filename
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded '{selected_paper['title']}' to {filename}")
        run_chatbot_with_document(filename)  # Assuming this function is defined to interact based on the document
    else:
        print("Failed to download the paper.")

if __name__ == "__main__":
    main()
