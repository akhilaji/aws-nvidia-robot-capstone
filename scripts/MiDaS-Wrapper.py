import os, sys
sys.path.append("/workspace/git/MiDaS/")
from run import run

arg_map = { 
    'i' : "/workspace/git/MiDaS/input",
    'o' : "/workspace/git/MiDaS/output",
    'm' : "/workspace/git/MiDaS/model-f46da743.pt"
}
for i in range(0, len(sys.argv)):
    if(sys.argv[i][0] == '-' and len(sys.argv[i]) > 1):
        arg_map[sys.argv[i][1]] = sys.argv[i + 1]
        i += 1
run(arg_map['i'], arg_map['o'], arg_map['m'])
