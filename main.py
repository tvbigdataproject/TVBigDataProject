import sys
import os
import argparse
import bigdata as bd
from pyspark.sql import SparkSession
from bigdata.RetweetTagsJaccardGraph import RetweetTagsJaccardGraph

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 pyspark-shell"
)

if __name__ == "__main__":
    """
    Processes a json file containing tweets and produces a unified graph containing retweets, 
    tags and Jaccard relationships.
    Existing files will be overwritten.
    
    Usage: main.py input_file [--output_path] [--id_neighbours id] [--save_full_graph] [--save_pbi_report] [--save_word_cloud] [--only_tags_from_not_retweetted_posts] 
    """

    # Creates a parser for the CLI arguments
    parser = argparse.ArgumentParser(description='Usage: main.py input_file [--output_path]\n'
                                                 'Processes a json file containing tweets and produces a unified graph containing retweets, '
                                                 'tags and Jaccard relationships.\n'
                                                 'Existing files will be overwritten.')
    parser.add_argument('input_file', type=str, help='Path to the input json file')
    parser.add_argument('--output_path', type=str, default=os.getcwd() + "\\outputs\\", help='Output path')
    parser.add_argument('--id_neighbours', type=str, help='Neighbours of id node')
    parser.add_argument('--save_full_graph', action=argparse.BooleanOptionalAction, default=False,
                        help='Save full graph')
    parser.add_argument('--save_pbi_report', action=argparse.BooleanOptionalAction, default=False,
                        help='Save Power BI report')

    parser.add_argument('--save_word_cloud', action=argparse.BooleanOptionalAction, default=False,
                        help='Save word cloud')
    parser.add_argument('--only_tags_from_not_retweetted_posts', action=argparse.BooleanOptionalAction, default=False,
                        help='Consider only tags from not retweeted posts')

    # Analyzes CLI arguments
    args = parser.parse_args()

    # Creates a SparkSession
    spark = SparkSession.builder.appName("BdProject").getOrCreate()
    print(f"Starting the processing of the file: {args.input_file}")

    # Builds full graph (retweet, hash, Jaccard)
    g = RetweetTagsJaccardGraph(spark, args.input_file, args.output_path,0.5,
                                args.only_tags_from_not_retweetted_posts)

    # Extracts text from tweets contents to build a word cloud
    if (args.save_word_cloud):
        print("Saving word cloud")
        bd.generateTextFromTweeter(spark, args.input_file, args.output_path)

    # Saves full graph
    if(args.save_full_graph):
        print("Saving full graph")
        g.saveFullGraph()

    # Saves Power BI report
    if(args.save_pbi_report):
        print("Saving Power BI report")
        g.exportReport()

    # Saves id's neighbours
    if(args.id_neighbours):
        print(f"Saving neighbours of node id={args.id_neighbours}")
        bd.saveGraph(g.extractNeighbours(args.id_neighbours), args.output_path + "\\id_neighbours_" + str(args.id_neighbours), "id")

    print(f"Elaboration completed.\n"
          f"Output path: {args.output_path}")

    spark.stop()