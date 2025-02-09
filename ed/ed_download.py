from edapi import EdAPI
import os
from dotenv import load_dotenv
import json
from tqdm import tqdm

# Find the root directory where .env is stored
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script's folder
root_dir = os.path.abspath(os.path.join(base_dir, ".."))  # Go up to the root folder


# Load .env from root
dotenv_path = os.path.join(root_dir, ".env")
load_dotenv(dotenv_path)


# initialize Ed API
ed = EdAPI()

# Read API token from environment
api_token = os.getenv("ED_API_TOKEN")

if not api_token:
    print("❌ ERROR: Missing ED_API_TOKEN. Set it in the .env file.")
    exit(1)

# Ensure output directory exists
output_dir = os.getenv("OUTPUT_DIR", "/app/threads")  # Default to /app/threads in Docker
json_output_dir = os.getenv("JSON_THREADS_DIR", "/app/json_threads")
raw_json_output_dir = os.getenv("RAW_JSON_THREADS_DIR", "/app/json_threads")
os.makedirs(output_dir, exist_ok=True)  # ✅ Creates directory if missing
os.makedirs(json_output_dir, exist_ok=True)  # ✅ Creates directory if missing

# Initialize EdAPI instance with token
ed = EdAPI()

# Manually set the token before making API calls
ed.token = api_token  # ✅ This is the correct way to set it

# Test authentication
try:
    user_info = ed.get_user_info()  # ✅ This should work correctly now
    print(f"✅ Authenticated as {user_info['user']['name']}")
except Exception as e:
    print(f"❌ ERROR: Authentication failed - {e}")
    exit(1)

try:
    year = os.getenv("ED_YEAR")
except:
    print("Add ED_YEAR variable to .env file")

# Get output directory from .env or default to "/app/threads"
output_dir = os.getenv("OUTPUT_DIR", "/app/threads")

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

def save_thread(thread_id, content):
    filename = os.path.join(output_dir, f"{thread_id}.txt")
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"Thread {thread_id} saved to {filename}")

def welcome():
    # retrieve user information; authentication is persisted to next API calls
    user_info = ed.get_user_info()
    user = user_info['user']
    print(f"Hello {user['name']}!")

def course_query(year=year):
    # Searches for course id to process
    #year = input('Year of Course: ')
    #semester = input("Spring, Summer, Fall? ")
    for course in [("ID: " +str(x['course']['id']), x['course']['name'], x['course']['year'], x['course']['session']) \
    for x in ed.get_user_info()['courses'] if (x['course']['year']==year)]:
        print(course)

def get_thread_ids(course_id):
    offset = 0
    limit = 100  # Adjust this if the API allows more per request
    all_threads = []

    print(f"Retrieving threads for Course ID: {course_id}")

    while True:
        # Retrieve a batch of threads
        threads = ed.list_threads(course_id=course_id, offset=offset, limit=limit)
        
        if not threads:
            print("✅ All threads retrieved.")
            break  # Exit loop when no more threads are returned
        
        print(f"Retrieved {len(threads)} threads (offset: {offset})")

        all_threads.extend(threads)  # Add threads to the list
        offset += limit  # Move to the next batch

    print(f"Total threads retrieved: {len(all_threads)}")
    return [thread['id'] for thread in all_threads]


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
    # Initialize tqdm to wrap the iterable
    for i, thread_id in enumerate(tqdm(thread_ids, desc="Processing Threads", unit="thread")):
        # Fetch the full thread content
        thread_content = ed.get_thread(thread_id)
        
        # Start the document with the parent thread content
        content = f"Thread ID: {thread_id}\n"
        content += f"User {thread_content['user_id']} (Parent Post) says:\n{thread_content['document']}\n\n"

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
            print("continue")  # Skip if 'anonymous_comments' is not a list

        # Save text to a file
        filename = os.path.join(output_dir, f"{thread_id}.txt")

        # Ensure the file is writable and remove if it exists
        if os.path.exists(filename):
            os.remove(filename)

        with open(filename, "w", encoding="utf-8") as file:
            file.write(content)

        # Save json to a file
        filename = os.path.join(json_output_dir, f"{thread_id}.json")

        # Ensure the file is writable and remove if it exists
        if os.path.exists(filename):
            os.remove(filename)
        
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(thread_content, file, indent=4)

def save_threads_as_json(thread_ids):
    """
    Fetches raw JSON for each thread and saves it locally.
    :param thread_ids: List of thread IDs to process
    """
    for thread_id in tqdm(thread_ids, desc="Downloading Threads", unit="thread"):
        try:
            # Fetch the raw thread JSON from the API
            thread_content = ed.get_thread(thread_id)
            
            # Define the file path
            filename = os.path.join(raw_json_output_dir, f"{thread_id}.json")
            
            # Ensure the file doesn't already exist
            if os.path.exists(filename):
                os.remove(filename)
            
            # Save the JSON to a file
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(thread_content, file, indent=4)
            
            print(f"✅ Thread {thread_id} saved to {filename}")
        
        except Exception as e:
            print(f"❌ Failed to fetch thread {thread_id}: {e}")

try:
    course_id = os.getenv("ED_COURSEID")
except:
    print("Add ED_COURSEID variable to .env file")

welcome()

course_query()

#Enter course ID 
#course_id = int(input("Enter ID of the Course from Above: "))
thread_ids = get_thread_ids(course_id)
try:
    doc_from_threads(thread_ids)
except:
    pass
try:    
    save_threads_as_json(thread_ids)
except:
    pass