from edapi import EdAPI
import os
from dotenv import load_dotenv
import time
import json

# Load environment variables from .env file
load_dotenv()

# initialize Ed API
ed = EdAPI()

# Read API token from environment
api_token = os.getenv("ED_API_TOKEN")

if not api_token:
    print("❌ ERROR: Missing ED_API_TOKEN. Set it in the .env file.")
    exit(1)

# Ensure output directory exists
output_dir = os.getenv("OUTPUT_DIR", "/app/threads")  # Default to /app/threads in Docker
os.makedirs(output_dir, exist_ok=True)  # ✅ Creates directory if missing

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
    limit = 1000  # Adjust based on API's pagination limit
    all_threads = []

    while True:
        threads = ed.list_threads(course_id=course_id, offset=offset, limit=limit)
        #for thread in threads:
            #print(f"Thread ID: {thread['id']}, Title: {thread.get('title')}") 
        if not threads:
            break  # No more threads to fetch
        all_threads.extend(threads)
        offset += limit  # Move to the next page

    thread_ids = [thread['id'] for thread in all_threads]
    return thread_ids


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
        
        filename = os.path.join(output_dir, f"raw_{thread_id}.json")
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(thread_content, file, indent=4)

        #print(f"Thread {thread_id} saved to {filename}")




try:
    course_id = os.getenv("ED_COURSEID")
except:
    print("Add ED_COURSEID variable to .env file")

welcome()

course_query()

#Enter course ID 
#course_id = int(input("Enter ID of the Course from Above: "))
thread_ids = get_thread_ids(course_id)

doc_from_threads(thread_ids)

