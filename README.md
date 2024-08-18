# Financial Investment Assistant Chatbot using Retrieval-Augmented Generation (RAG)

**Purpose**: This project leverages Retrieval-Augmented Generation (RAG) to create a conversational chatbot with memory, that assists users with financial investment decisions. The chatbot can recall previous interactions within a session, providing a more personalized and coherent user experience.

**Data Sources**: The chatbot uses publicly available reports from leading Vietnamese securities firms, embedded into a vector database using OpenAI's embedding model. This ensures that the chatbot provides high-quality, localized financial insights tailored to the Vietnamese market

This chatbot is web-based and created using Streamlit, making it easy to deploy and interact with.

## Getting Started

To run the chatbot on your local machine, follow these steps:

1. **Install Dependencies**:
   Ensure you have Python installed, then install the required packages by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Chatbot**:
   Open your terminal and execute the following commands:

   ```bash
   # Navigate to the chatbot directory
   cd chatbot

   # Run the Streamlit app
   streamlit run chatbot.py
   ```

## Demo
https://github.com/user-attachments/assets/2f4c74b9-e592-47a2-b8ff-ef4946bd5425
