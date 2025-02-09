import chainlit as cl
from helpers.rag_helper import generate_rag_response

# Set up clickable starters
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Course Project Ideas",
            message="Where can I find a team? What other projects have people done in the past? What kind of project might be impactful and utilize the topics in this course?",
            #icon="/public/idea.svg"
        ),
        cl.Starter(
            label="Explain linear algebra for this course",
            message="Explain linear algebra for computing machine learning like I'm an undergraduate with a grasp on math up to and including integral calculus but little more.",
            #icon="/public/learn.svg"
        ),
        cl.Starter(
            label="Python script for my homework",
            message="Can I use generated code for my homework?",
            #icon="/public/terminal.svg"
        ),
        cl.Starter(
            label="Submitting homework",
            message="How should my homework be submitted? Do I need Gradescope? Are multiple files required? How to format the files?",
            #icon="/public/write.svg"
        )
    ]

# Welcome message at the start of the chat
#@cl.on_chat_start
#async def on_chat_start():
#    await cl.Message(content="Welcome to ISYE6740! What's on your mind?").send()

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