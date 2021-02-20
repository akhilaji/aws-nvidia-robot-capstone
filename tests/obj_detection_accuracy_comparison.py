import sys
import pandas as pd

if len(sys.argv) == 3:
    # read the two files into data frames
    base = pd.read_csv(sys.argv[1])
    test = pd.read_csv(sys.argv[2])

    entries = min(len(base['frame']), len(test['frame']))

    unmatched_frames = 0
    unmatched_classes = 0

    base_i = 0
    test_i = 0

    percent = 0
    for i in range(entries):
        if base['frame'][base_i] == test['frame'][base_i]:
            if base['class_id'][i] == test['class_id'][test_i]:
                percent = percent + (test['conf'][test_i] / base['conf'][base_i]) * 100

                base_i += 1
                test_i += 1
            else:
                while base['class_id'][base_i] != test['class_id'][test_i]:
                    base_i += 1

                    unmatched_classes += 1

                    if base['frame'][base_i] != test['frame'][test_i]:
                        break
        else:
            while base['frame'][base_i] < test['frame'][test_i]:
                base_i += 1
                unmatched_frames += 1

            while test['frame'][test_i] < base['frame'][base_i]:
                test_i += 1
                unmatched_frames += 1
    
    percent = percent / entries

    print("Umatched Frames: {}".format(unmatched_frames))
    print("Umatched Classes: {}".format(unmatched_classes))

    print("Net Percentage Increase: {}%".format(percent))
else:
    print("Insuffucient arguments supplied")
