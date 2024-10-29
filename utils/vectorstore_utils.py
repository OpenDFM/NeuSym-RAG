#coding=utf8
import os, sys, logging
from typing import List, Dict, Any, Union, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectorstore', type=str, help='which vectorstore to use')
    parser.add_argument('--embed_model', type=str, default='', help='which embedding model to use, you can download the model in advance into cache folder ./.cache/')
    parser.add_argument('--')