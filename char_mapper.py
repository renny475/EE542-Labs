#!/usr/bin/env python3
import sys

for line in sys.stdin:
    for char in line.rstrip("\n"):
        if char == " ":
            char = "<SPACE>"
        elif char == "\t":
            char = "<TAB>"

        print(f"{char}\t1")
        
