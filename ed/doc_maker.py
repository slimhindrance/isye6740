from edapi import EdAPI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize Ed API
ed = EdAPI()
ed.login()  # Authenticate user through the ED_API_TOKEN

def extract_all_nested_comments(thread_content):
    """
    Recursively extracts all comments (including nested ones) from a thread.
    Handles both 'comments' and 'anonymous_comments'.
    :param thread_content: The full thread content from ed.get_thread()
    :return: List of all comments
    """
    all_comments = []

    def recursive_extract(comment, is_anonymous=False):
        # Add the current comment
        if is_anonymous:
            comment['user_id'] = "ANON"  # Mark anonymous user
        all_comments.append(comment)

        # Recurse into nested comments if any
        for sub_comment in comment.get('comments', []):
            recursive_extract(sub_comment, is_anonymous=is_anonymous)

    # Process regular comments
    for comment in thread_content.get('comments', []):
        recursive_extract(comment, is_anonymous=False)

    # Check if 'anonymous_comments' is a list
    anon_comments = thread_content.get('anonymous_comments', [])
    if isinstance(anon_comments, list):
        for anon_comment in anon_comments:
            recursive_extract(anon_comment, is_anonymous=True)
    else:
        print(f"Skipping 'anonymous_comments' for thread {thread_content['id']} as it is not a list.")

    return all_comments

def doc_from_threads(thread_ids):
    """
    Creates text documents for each thread, including parent thread content, answers, and all comments.
    The output is structured to match the natural flow of the thread.
    :param thread_ids: List of thread IDs to process
    """
    for thread_id in thread_ids:
        time.sleep(.2)
        # Fetch the full thread content
        thread_content = ed.get_thread(thread_id)

        # Start the document with the parent thread content
        content = f"Thread ID: {thread_id}\n"
        content += f"User {thread_content['user_id']} (Parent Post) says:\n{thread_content['content']}\n\n"

        # Process all answers
        for answer in thread_content.get('answers', []):
            content += f"Answer from User {answer['user_id']}:\n{answer['document']}\n\n"
            # Include nested comments for each answer
            for sub_comment in answer.get('comments', []):
                content += f"  Comment from User {sub_comment['user_id']}:\n  {sub_comment['document']}\n\n"

        # Process regular comments
        for comment in thread_content.get('comments', []):
            content += f"Comment from User {comment['user_id']}:\n{comment['document']}\n\n"
            # Include nested comments
            for sub_comment in comment.get('comments', []):
                content += f"  Reply from User {sub_comment['user_id']}:\n  {sub_comment['document']}\n\n"

        # Process anonymous comments
        anon_comments = thread_content.get('anonymous_comments', [])
        if isinstance(anon_comments, list):
            for anon_comment in anon_comments:
                content += f"Anonymous Comment (User ANON):\n{anon_comment['document']}\n\n"
                # Include nested anonymous comments
                for sub_comment in anon_comment.get('comments', []):
                    content += f"  Reply from User ANON:\n  {sub_comment['document']}\n\n"
        else:
            print(f"Skipping 'anonymous_comments' for thread {thread_id} as it is not a list.")

        # Save to a file
        filename = f"/home/lindeman/isye6740/ed/threads/{thread_id}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(content)

        print(f"Thread {thread_id} saved to {filename}")

