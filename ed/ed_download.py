from edapi import EdAPI
import dotenv, boto3
from botocore.exceptions import NoCredentialsError
import os
import boto3
import os
from dotenv import load_dotenv
import doc_maker, ed_helper


# Load environment variables from .env file
load_dotenv()

ed_helper.welcome()

ed_helper.course_query()

#Enter course ID 
course_id = 70719#int(input("Enter ID of the Course from Above: "))
thread_ids = ed_helper.get_thread_ids(course_id)

doc_maker.doc_from_threads(thread_ids)

