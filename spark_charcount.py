#!/usr/bin/env python3
from pyspark import SparkContext

sc = SparkContext.getOrCreate()

text = sc.textFile(
    "hdfs://172.31.12.140:9000/gutenberg/*.txt"
)

def label_character(char):
    if char == " ":
        return "<SPACE>"
    if char == "\t":
        return "<TAB>"
    return char

counts = (
    text.flatMap(lambda line: list(line))
        .map(lambda char: (label_character(char), 1))
        .reduceByKey(lambda a, b: a + b)
)

counts.map(
    lambda item: f"{item[0]}\t{item[1]}"
).saveAsTextFile(
    "hdfs://172.31.12.140:9000/output_spark_charcount_1node"
)

sc.stop()
