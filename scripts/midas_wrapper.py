import argparse
import os
import sys

if __name__ == "__main__":
    repo_path = os.path.join('workspace','git','MiDaS')
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input',
        default=os.path.join(repo_path, 'input'),
        help='path to a folder of images')
    arg_parser.add_argument('-o', '--output',
        default=os.path.join(repo_path, 'output'),
        help='path to output folder')
    arg_parser.add_argument('-m', '--model',
        default=os.path.join('workspace', 'models', 'intel-isl_MiDaS_v2_model-f46da743.pt'),
        help='path to model')
    namespace, args = arg_parser.parse_known_args(sys.argv)
    try:
        sys.path.append(repo_path)
        from run import run
        if(not os.path.exists(namespace.output)):
            os.makedirs(namespace.output)
        run(namespace.input, namespace.output, namespace.model)
    except ImportError as ie:
        print(ie)
        exit(1)