import inotify.adapters
import os
import subprocess

notifier = inotify.adapters.Inotify()
notifier.add_watch('../video/')

for event in notifier.event_gen():
    if event is not None:
        #print event
        if 'IN_CREATE' in event[1]:
            print("file " + event[3] + " created in " + event[2])
            #Call Any Scripts Here
            #os.system("script_name.py")
            #to run multiple scripts:
            #subprocess.run("python3 script1.py & python3 script2.py", shell=True)
