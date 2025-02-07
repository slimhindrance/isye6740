from langchain_community.document_loaders import JSONLoader
import json, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
input_dir = os.getenv("INPUT_DIR")
print(input_dir)
filenames = [x for x in os.listdir(input_dir) if x[-4:]=="json"]

#print(filenames)


file_path = os.path.join(input_dir, filenames[-1])
data = json.loads(Path(file_path).read_text())
loader = JSONLoader(
         file_path=file_path,
         jq_schema='[.document]+[.comments[].document]',
         text_content=False)

docs = loader.load()
print(docs[0].page_content[:100])
print(docs[0].metadata)