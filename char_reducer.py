#!/usr/bin/env python3
import sys
from collections import defaultdict

counts = defaultdict(int)

for line in sys.stdin:
    char, value = line.rstrip("\n").split("\t", 1)
    counts[char] += int(value)

for char, count in counts.items():
    print(f"{char}\t{count}")
    
