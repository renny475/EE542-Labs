#!/usr/bin/env python3
from pyspark import SparkContext

sc = SparkContext.getOrCreate()

text = sc.textFile(
    "hdfs://172.31.12.140:9000/gutenberg/*.txt"
)

counts = (
    text.flatMap(lambda line: line.split())
        .map(lambda word: (word, 1))
        .reduceByKey(lambda a, b: a + b)
        .cache()
)

min_word = counts.min(key=lambda item: item[1])
max_word = counts.max(key=lambda item: item[1])

print("MIN:", min_word)
print("MAX:", max_word)

sc.stop()

