import chainlit as cl
from helpers.rag_helper import generate_rag_response

# Set up starters before chat starts
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Course Project Ideas",
            message="Where can I find a team? What other projects have people done in the past? What kind of project might be impactful and utilize the topics in this course?",
            icon="/public/idea.svg",
        ),
        cl.Starter(
            label="Explain linear algebra for this course",
            message="Explain linear algebra for computing machine learning like I'm five years old.",
            icon="/public/learn.svg",
        ),
        cl.Starter(
            label="Python script for daily email reports",
            message="Warn me against submitting generated code, as it violates student integrity and may be cause for dismissal. Provide links to Python and machine learning education resources",
            icon="/public/terminal.svg",
        ),
        cl.Starter(
            label="Format my document text into Latex",
            message="Ask me to paste in my report, which you will convert into Latex code for rendering",
            icon="/public/write.svg",
        )
    ]

# Initialize Chainlit app
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Welcome to ISYE6740! What's on your mind?").send()

# Handle user messages
@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content
    
    try:
        # Generate response using RAG
        response = generate_rag_response(user_input)
        await cl.Message(content=response).send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()

# Optional: Run Chainlit on 0.0.0.0:8888
if __name__ == "__main__":
    import os
    os.system("chainlit run chainlit_app.py --host 0.0.0.0 --port 8888")