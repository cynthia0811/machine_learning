import sys
sys.path.append("./py_basic/")
from event.send import Sender
from event.push import msg


def main():

    s = Sender()
    s.log("Core > Event > Send")

    msg.output()

main()