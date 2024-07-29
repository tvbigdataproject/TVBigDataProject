from pyspark.sql import functions as F
from pyspark.sql.functions import array_distinct, split, array_sort, col, lower, when, udf
from pyspark.sql.types import ArrayType, StringType
import spacy #python -m spacy download en_core_web_sm
from itertools import chain


"""
loads trained pipelines to perform tweets data cleaning 
"""
nlp = spacy.load('en_core_web_sm')

"""
User Defined Functions - UDF

array_to_string_udf: converts array of strings into single string separated by ","
flatten: aggregation function aggregate multiple arrays into one array of strings
jaccard: computes jaccard similarity function (intersection over union)
cntElemts: returns the number of elements in an array
cntIntersection: returns the number of elements in the intersection of two arrays
cleanerUDF: cleans the input string, eliminates not alphanumeric characters
"""
array_to_string_udf = F.udf(lambda x: '[' + ','.join([str(elem) for elem in x]) + ']')
flatten = udf(lambda x: list(chain.from_iterable(x)), ArrayType(StringType()))
jaccard = F.udf(lambda x, y: float(len(set(x).intersection(y))) / float(len(set(x).union(y))))
cntElemts = F.udf(lambda x: len(set(x)))
cntIntersection = F.udf(lambda x, y: len(set(x).intersection(y)))
cleanerUDF = udf(lambda x: cleaner(x), StringType())


def cleaner(string):
    """
    Cleans the input string, eliminates not alphanumeric characters

    @param string: input string
    @type string:
    @return: cleaned string separated by ","
    @rtype: string
    """
    if(string is None):
        return ""
        # Generate list of tokens
    doc = nlp(string)
    lemmas = [token.lemma_ for token in doc]  # Remove tokens that are not alphabetic
    a_lemmas = [lemma for lemma in lemmas if
                    lemma.isalpha() or lemma == '-PRON-']  # Print string after text cleaning
    return ' '.join(a_lemmas)

def nomalizeHashTags(df, hts):
    """
    Normalizes the text: lowercase, eliminates accented characters etc., removes duplicated hashtag, orders the results

    @param df: input dataframe
    @type df: DataFrame
    @param hts: column names of hashtags
    @type hts: string
    @return: dataframe with normalized hashtags
    @rtype: DataFrame
    """
    df = (df.withColumn('norm',
                        F.translate(
                            lower(
                                F.array_join(F.col(hts), ",")), 'ãäöüẞáäčďéěíĺľňóôŕšťúùůýž', 'aaousaacdeeillnoorstuuuyz')))
    df = df.withColumn("no_dupes_ordered",
                       array_sort(
                           array_distinct(
                               split("norm", ","))))
    #drop columns hts, norm, rename no_dupes_ordered to hts
    df = df.drop(col("hts"), col("norm")).withColumnRenamed("no_dupes_ordered", "hts",)
    return df


def generateTextFromTweeter(sparkSession, twJsonFilename, csvPath):
    """
    Extracts cleaned txt from tweets in order to build a wordcloud
        if the post is a RT, RT text field is considered, otherwise text field is considered

    @param twJsonFilename: input tweet json filename
    @type twJsonFilename: string
    @param csvPath: csv filename to write to
    @type csvPath: string
    @param sparkSession: spark session
    @type sparkSession: SparkSession
    """
    tw = sparkSession.read.json(twJsonFilename)

    twc = tw.select(when(
        col("retweeted_status.text").isNull(),
        cleanerUDF(col("text"))).otherwise(
        cleanerUDF(col("retweeted_status.text"))).alias("txt_plus_rt"))
    twc.repartition(1).write.mode("overwrite").csv(
        csvPath + "wordCloud",
        header=True)
    return None

def saveGraph(graph, path, prefix):
    """
    Save a GraphFrames graph into two files: vertices and edges

    @param graph: a GraphFrames graph
    @type graph: GraphFrames
    @param path: path to save the graph
    @type path: string
    @param prefix: prefix to save the graph (saved files - prefix.edges.csv, prefix.vertices.csv)
    @type prefix: string
    """
    graph.edges.repartition(1).write.mode("overwrite").csv(
        path + "\\" + prefix + ".edges.csv",
        header=True)

    graph.vertices.repartition(1).write.mode("overwrite").csv(
        path + "\\" + prefix + ".vertices.csv",
        header=True)
    return None