from edapi import EdAPI
from dotenv import load_dotenv
import os

load_dotenv()

# initialize Ed API
ed = EdAPI()
# authenticate user through the ED_API_TOKEN environment variable
ed.login()

try:
    year = os.getenv("ED_YEAR")
except:
    print("Add ED_YEAR variable to .env file")

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
