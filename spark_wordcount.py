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
)

counts.saveAsTextFile(
    "hdfs://172.31.12.140:9000/output_spark_wordcount_1node"
)

sc.stop()

