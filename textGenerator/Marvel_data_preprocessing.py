import os

current_directory = os.getcwd()

marvel_directory = os.path.join(current_directory, 'marvel')

concatenated_text = ''

for filename in os.listdir(marvel_directory):
    file_path = os.path.join(marvel_directory, filename)
    
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            file_contents = file.read()
        
        concatenated_text += file_contents

output_file_path = os.path.join(current_directory, 'marvel.txt')
with open(output_file_path, 'w') as output_file:
    output_file.write(concatenated_text)
