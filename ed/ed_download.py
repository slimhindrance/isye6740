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

# Retrieve AWS credentials from environment variables
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

ed_helper.welcome()

ed_helper.course_query()

#Enter course ID 
course_id = int(input("Enter ID of the Course from Above: "))
thread_ids = ed_helper.get_thread_ids(course_id)

doc_maker.doc_from_threads(thread_ids)


"""
# Initialize S3 client
s3_client = aws_helper.get_client('s3', aws_access_key, aws_secret_key, aws_region)

# Upload the documents to s3
for filename in os.listdir('threads'):
    aws_helper.upload_file_to_s3(f"threads/{filename}", 'isye6740', s3_client, filename)
"""

