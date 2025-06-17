import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from run_x_learner import run_custom_x_learner

def main():
    print("--- Running Custom X-Learner ---")
    run_custom_x_learner()

if __name__ == "__main__":
    main()