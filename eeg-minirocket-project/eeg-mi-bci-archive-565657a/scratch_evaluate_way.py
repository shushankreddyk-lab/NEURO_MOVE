import sys
sys.path.insert(0, r"D:\pip_packages")
import os
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from evaluate_all_models import evaluate_models

def main():
    evaluate_models('models')

if __name__ == "__main__":
    main()
