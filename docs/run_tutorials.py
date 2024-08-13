import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys
import shutil

def run_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
            with open(notebook_path, 'wt') as f:
                nbformat.write(nb, f)
            print(f"Executed: {notebook_path}")
            print(f"Output: {notebook_path}")
        except Exception as e:
            print(f"Error executing the notebook {notebook_path}: {e}")

def run_all_notebooks(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(root, file)
                print(f"Running notebook: {notebook_path}")
                run_notebook(notebook_path)

def copy_folder(origen, destino):
    try:
        if not os.path.exists(destino):
            os.makedirs(destino)
        shutil.copytree(origen, destino, dirs_exist_ok=True)
        #run_all_notebooks(destino)
        print(f"Folder copied from {origen} to {destino} sucessfully.")
    except Exception as e:
        print(f"Error when trying to copy folder: {e}")


if __name__ == '__main__':
    # Run all the tutorials
    work_dir = os.getcwd()
    src_path = os.path.join(work_dir,'src')
    tutorials_path = os.path.join(work_dir,'tutorials')
    docs_path = os.path.join(work_dir,'docs','source')
    sys.path.insert(0, src_path)

    verticals = ['bias', 'datasets', 'explainability', 'robustness', 'security']  

    for vertical in verticals:
        path = os.path.join(tutorials_path, vertical)
        copy_folder(path, os.path.join(docs_path, 'gallery', 'tutorials', vertical))