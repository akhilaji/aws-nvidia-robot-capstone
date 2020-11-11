import argparse
import os
import sys

if(__name__ == "__main__"):
    repo_path = os.path.join(os.path.sep + 'workspace','git','MiDaS')
    default_input = os.path.join(repo_path, 'input')
    default_output = os.path.join(repo_path, 'output')
    default_model = os.path.join(os.path.sep + 'workspace', 'models', 'intel-isl_MiDaS_v2_model-f46da743.pt')

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input',  default = default_input,  help = 'path to a folder of images')
    arg_parser.add_argument('-o', '--output', default = default_output, help = 'path to output folder')
    arg_parser.add_argument('-m', '--model',  default = default_model,  help = 'path to model')

    ns, args = arg_parser.parse_known_args(sys.argv)
    try:
        sys.path.append(repo_path)
        from run import run
        if(not os.path.exists(ns.output)):
            os.makedirs(ns.output)
        run(ns.input, ns.output, ns.model)
    except ImportError as ie:
        print(ie)