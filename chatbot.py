from transformers import AutoTokenizer, AutoModelForCausalLM,TextStreamer
import fitz  # PyMuPDF for handling PDF files

def run_chatbot_with_document(filename):
    # Initialize the Zephyr model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HTomGrc/FusionNet_7Bx2_MoE_14B")
    model = AutoModelForCausalLM.from_pretrained("TomGrc/FusionNet_7Bx2_MoE_14B",load_in_4bit=True,device_map='cuda:0')
    
    # Function to convert PDF document to text
    def pdf_to_text(filename):
        doc = fitz.open(filename)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    # Function to generate an answer using Zephyr, keeping track of the context
    def generate_answer(question, context, tokenizer, model, max_length=8192):
        # Combine the context with the new question
        role = """You are an assistant who accurately Answers based on Question and Context purely according to the Document. Only according to Question"""
        prompt = f"###role: {role}\n\n ###Document: {document} \n\n ###Context: {context}\n\n ###Question: {question}\n\n Answers: "
        
        # Generate the response
        inputs = tokenizer.encode(prompt, return_tensors='pt') #, max_length=max_length, truncation=True)
        streamer = TextStreamer(tokenizer)
        outputs = model.generate(inputs, max_new_tokens=max_length,streamer = streamer ,temperature = 0.3, top_p = 0.3,top_k = 45)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer from the generated text
        answer = answer.split("Answers:")[1].strip()
        return answer

    # Convert the PDF to text
    document_text = pdf_to_text(filename)

    # Main chat interface
    document = f"Document: {document_text[:4000]}"  # Initial context from the document, adjust as needed
    context = ""
    print("Document-based Chatbot initialized. Ask me anything about the document! Type 'quit' to exit.")

    while True:
        user_question = input("You: ")
        if user_question.lower() == "quit":
            print("Chatbot terminated.")
            break

        answer = generate_answer(user_question, context, tokenizer, model)
        print(f"Bot: {answer}")
        # Update context with the latest Q&A
        context += f"\nQuestion: {user_question}\nAnswer: {answer}"
        
        # Check total token count, and if it exceeds 5000, remove the oldest 2000 tokens
        context_tokens = tokenizer.tokenize(context)
        if len(context_tokens) > 3000:
            # Keep the most recent 3000 tokens, adjust numbers as needed
            context = tokenizer.decode(context_tokens[-3000:])