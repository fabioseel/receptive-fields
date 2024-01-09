import os
import fnmatch

def find_files_in_folder(folder, partial_name):
    matching_files = []
    
    for root, dirs, files in os.walk(folder):
        for filename in fnmatch.filter(files, f'*{partial_name}*'):
            matching_files.append(os.path.join(root, filename))
    matching_files.sort()
    return matching_files