import os
import re

thread_dir =  "ed/threads"
clean_dir = "ed/clean_threads"

pattern = '\<.*?\>'

for filename in os.listdir('ed/threads'):
    print(filename)
    
    with open(f"{thread_dir}/{filename}", 'r') as file:
        content = file.read()
        new = re.sub(pattern, " ", content)
        
        with open(f"{clean_dir}/{filename}", "w") as cleanfile:
            cleanfile.write(new)