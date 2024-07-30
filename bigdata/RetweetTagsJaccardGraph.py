import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import explode, collect_set, col, lit, concat, array_except, array
from graphframes import GraphFrame

from bigdata import nomalizeHashTags, saveGraph, array_to_string_udf, flatten, jaccard, cntElemts, cntIntersection

"""
The class handles json tweets (see the schema schema_twitter.txt in the main folder) 
"""
class RetweetTagsJaccardGraph:

    def __init__(self, sparkSession, twFile, bp=os.getcwd(), jaccard_threshold=0.5,
                 onlyTagsFromNotRetweetedPosts=False):
        """
        Constructor for RetweetTagsJaccardGraph

        @param twFile: tweets filename (path must be included) to elaborate
        @type twFile: string
        @param sparkSession: spark session
        @type sparkSession: SparkSession
        @param bp: optional base path used to write to (default is the working directory)
        @type bp: string
        @param jaccard_threshold: Jaccard threshold to consider (default value is greater than 0.5)
        @type jaccard_threshold: float
        @param onlyTagsFromNotRetweetedPosts: flag - if true it considers only tweets from not-retweeted posts
        @type onlyTagsFromNotRetweetedPosts: boolean
        """
        #init fields of the class
        self.basePath = bp
        self.twitterFile = twFile
        self.sparkSession = sparkSession
        self.tw = self.sparkSession.read.json(self.twitterFile)
        self.jc_threshold = jaccard_threshold

        #generateRetweetTagsJaccardGraph
        self.df_rt_grouped, self.u_rt = self.generateRetweetGraph()
        self.df_ht_grouped, self.u_ht, self.user_hts_mapping = self.generateHashtagsGraph(onlyTagsFromNotRetweetedPosts)
        v = self.u_rt.union(self.u_ht).distinct()
        e = self.df_rt_grouped.union(self.df_ht_grouped)
        # first it generates the g graph, than it uses motifs to find couples of users having at least two hashtags in common
        g = GraphFrame(v, e)
        self.df_jc, self.u_jc = self.generateJaccardGraph(g, self.user_hts_mapping)
        v = v.union(self.u_jc).distinct()
        e = e.union(self.df_jc).distinct()
        self.gFull = GraphFrame(v, e)

    def generateRetweetGraph(self):
        """
        Generates the retweet graph (creates an edge a->b if "a" retweeted at least one post of "b")

        @return: tuple
                 df_rt_grouped: dataframe containing the retweet graph (src, dst, w, type=RT)
                                w stores the number of times "b" has been retweeted by "a"
                 u_rt: dataframe containing the list of distinct users (vertices) of the graph (id)
        @rtype: tuple
                 df_rt_grouped: dataframe
                 u_rt: dataframe
        """
        #filters only retweet posts
        df_rt = self.tw.filter(col("retweeted_status").isNotNull())
        #constructs the edges dataframe
        df_rt = df_rt.select(col("retweeted_status.user.id").alias("src"), col("user.id").alias("dst"))
        df_rt_grouped = df_rt.groupBy(col("src"), col("dst")).count()
        df_rt_grouped = df_rt_grouped.withColumnRenamed("count", "w")
        df_rt_grouped = df_rt_grouped.withColumn("type", lit("RT"))
        #constructs the vertices dataframe
        src_rt = df_rt_grouped.select("src")
        dst_rt = df_rt_grouped.select("dst")
        u_rt = src_rt.union(dst_rt).distinct().withColumnRenamed("src", "id")

        return df_rt_grouped, u_rt

    def generateHashtagsGraph(self, onlyTagsFromNotRetweetedPosts):
        """
        Generates the graph (creates an edge a->b if "b" is a hashtag used by "b" in some "a" post)

        @param onlyTagsFromNotRetweetedPosts: flag - if true it considers only tweets from not-retweeted posts
        @type onlyTagsFromNotRetweetedPosts: boolean
        @return: tuple
                 df_ht_grouped: dataframe containing the hashtags graph (src, dst, w, type=HT)
                                w stores the number of times hashtag "b" is used by "a"
                 u_ht: dataframe contains the list of distinct users and hashtags (vertices) of the graph (id)
                 user_hts_mapping: dataframe contains for each user the list of the used hashtags
        @rtype: tuple
                 df_ht_grouped: dataframe
                 u_ht: dataframe
                 user_hts_mapping: dataframe
        """
        # filter RT hashtags
        df_ht_rt = (self.tw.
                    filter("retweeted_status.user.id is NOT NULL").
                    filter("retweeted_status.hashtagEntities is NOT NULL").
                    select(col("retweeted_status.user.id").alias("id"),
                           col("retweeted_status.hashtagEntitiesArray").alias("htea")))

        # filters normal post hashtags without RT
        # if onlyTagsFromNotRetweetedPosts=True excludes RT because are considered above and it discards the "main" post
        # if onlyTagsFromNotRetweetedPosts=False it considers both hashtags of post and retweet
        filter = "TRUE"
        if (onlyTagsFromNotRetweetedPosts):
            filter = "retweeted_status.user.id is NULL"

        df_ht_tw = (self.tw.
                    filter(filter).
                    filter("user.id is not NULL").
                    filter("hashtagEntities is not NULL").
                    select(col("user.id").alias("id"), col("hashtagEntitiesArray").alias("htea")))

        #joins of the two sets filtered above
        df_ht = df_ht_tw.union(df_ht_rt)
        #stores all hashtags in hts field
        df_ht = df_ht.groupBy(col("id")).agg(flatten(collect_set("htea")).alias("hts"))
        #lowercase, remove accented characters, remove duplicates, order
        df_ht = nomalizeHashTags(df_ht, "hts")

        user_hts_mapping = df_ht
        #creates one edge for every tag in the array
        df_ht = df_ht.withColumn("htsExplode", explode(df_ht["hts"]))
        df_ht = df_ht.withColumn("ht", df_ht["htsExplode"]).select(col("id").alias("src"), col("ht").alias("dst"),
                                                                   lit("HT").alias("type"))
        #counts how many times the hashtag is present in a post or in a retweet of a specific user (src)
        df_ht_grouped = df_ht.groupBy(col("dst"),
                                      col("src")).count()
        #reorders columns and renames
        df_ht_grouped = (df_ht_grouped.withColumnRenamed("count", "w")
                         .select(col("src"), col("dst"), col("w")))
        df_ht_grouped = df_ht_grouped.withColumn("type", lit("HT"))

        #estracts vertices
        src_ht = df_ht_grouped.select(col("src").alias("id"))
        dst_ht = df_ht_grouped.select(col("dst").alias("id"))
        u_ht = src_ht.union(dst_ht).distinct()
        return df_ht_grouped, u_ht, user_hts_mapping

    def generateJaccardGraph(self, g, user_hts_mapping):
        """
        Generates the graph of Jaccard similarity between a couple of users:
            Creates an edge a<->b if
             * "a" and "b" have at least two hashtags in common
             * "a" and "b" have Jaccard hashtags similarity > threshold

        @param g: tweets hashtag graph
        @type g: GraphFrames graph
        @param user_hts_mapping: dataframe contains for each user the list of the used hashtags
        @type user_hts_mapping: dataframe
        @return: tuple
                 df_jc: dataframe contains the edges of Jaccard similarity between a couple of users (src, dst, w, type=JC)
                                w stores the Jaccard similarity between src and dst
                 u_jc: dataframe contains the list of distinct users (vertices) of the graph (id)
        @rtype: tuple
                 df_jc: dataframe
                 u_jc: dataframe
        """

        #searches for couple of users that shares at least two hashtags
        #(a.id > b.id) and (c.id > d.id) avoids duplicates due to simmetric relationship
        motifsTwoUsersTwoHashTags = g.find("(c)-[e]->(a); (d)-[e1]->(a); (c)-[e2]->(b); (d)-[e3]->(b)")
        couplesToCompare = motifsTwoUsersTwoHashTags.filter(
            "e.type = 'HT' and "
            "e1.type = 'HT' and "
            "e2.type = 'HT' and "
            "e3.type = 'HT' and "
            "(a.id > b.id) and "
            "(c.id > d.id)").select(col("c"), col("d"))

        #adds hashtag lists to the couples
        c1 = couplesToCompare.join(user_hts_mapping, (couplesToCompare["c.id"] == user_hts_mapping["id"]))
        c1 = c1.drop(col("c")).withColumnRenamed("id", "id1").withColumnRenamed("hts", "hts1")

        c2 = c1.join(user_hts_mapping, (c1["d.id"] == user_hts_mapping["id"]))
        c2 = c2.drop(col("d")).withColumnRenamed("id", "id2").withColumnRenamed("hts", "hts2")

        #calculates Jaccard similarity
        c2 = c2.select(col("id1"), F.concat_ws(',', col("hts1")).alias("hts1"),
                       col("id2"), F.concat_ws(',', col("hts2")).alias("hts2"),
                       jaccard(col("hts1"), col("hts2")).alias("jc"),
                       cntElemts(col("hts1")).alias("nt1"),
                       cntElemts(col("hts2")).alias("nt2"),
                       cntIntersection(col("hts1"), col("hts2")).alias("intersection"))

        #filters low similarity
        df_jc = c2.filter(col("jc") > self.jc_threshold)
        df_jc = df_jc.select(col("id1").alias("src"), col("id2").alias("dst"), col("jc").alias("w"),
                             lit("JC").alias("type"))

        #extracs edges (users)
        src_jc = df_jc.select("src")
        dst_jc = df_jc.select("dst")
        u_jc = src_jc.union(dst_jc).distinct().withColumnRenamed("src", "id")

        return df_jc, u_jc

    def exportReport(self):
        """
        Generates summary report for power BI
        for each user "ut" it reports:
            * the list of hashtags used in all the post,
            * the list of users that "ut" has retweeted,
            * the list of users that retweet "ut" posts,
            * the list of users having Jaccard similarity to "ut" greater than threshold
        fields: ut | tags | RT list | be RT list | JC list

        """

        # adds retweet info (users retweeted by user "ut")
        user_hts_RT = self.user_hts_mapping.join(self.df_rt_grouped,
                                                 (self.user_hts_mapping["id"] == self.df_rt_grouped["src"]), "left")
        user_hts_RT = user_hts_RT.drop("src", "w", "type")
        user_hts_RT = user_hts_RT.withColumnRenamed("id", "user")
        user_hts_RT = user_hts_RT.withColumnRenamed("dst", "retweeted_users")
        user_hts_RT = user_hts_RT.groupBy(col("user"), col("hts")).agg(
            collect_set("retweeted_users").alias("rt_users"))

        # adds be rt
        # adds users retweet info (users that retweeted user "ut")
        user_hts_RT_be_RT = user_hts_RT.join(self.df_rt_grouped, (user_hts_RT["user"] == self.df_rt_grouped["dst"]),
                                             "left")
        user_hts_RT_be_RT = user_hts_RT_be_RT.drop("dst", "w", "type")
        user_hts_RT_be_RT = user_hts_RT_be_RT.withColumnRenamed("src", "be_retweeted_users")
        user_hts_RT_be_RT = user_hts_RT_be_RT.groupBy(col("user"), col("hts"), col("rt_users")).agg(
            collect_set("be_retweeted_users").alias("bert_users"))

        #adds Jaccard similarity users (jc > threshold in the graph)
        user_hts_RT_be_RT_JC = user_hts_RT_be_RT.join(self.df_jc,
                                                      (user_hts_RT_be_RT["user"] == self.df_jc["dst"]) |
                                                      (user_hts_RT_be_RT["user"] == self.df_jc["src"]), "left")
        user_hts_RT_be_RT_JC = user_hts_RT_be_RT_JC.drop("w", "type")
        user_hts_RT_be_RT_JC = user_hts_RT_be_RT_JC.groupBy(col("user"), col("hts"), col("rt_users"),
                                                            col("bert_users")).agg(
            array_except(
                concat(collect_set("src").alias("jc_src"), collect_set("dst").alias("jc_dst")),
                array(col("user"))
            ).alias("jc_users")
        )
        #converts array to string
        user_hts_RT_be_RT_JC = user_hts_RT_be_RT_JC.withColumn('hashTags', array_to_string_udf(col("hts")))
        user_hts_RT_be_RT_JC = user_hts_RT_be_RT_JC.withColumn('retweetUsers', array_to_string_udf(col("rt_users")))
        user_hts_RT_be_RT_JC = user_hts_RT_be_RT_JC.withColumn('beRetweetUsers', array_to_string_udf(col("bert_users")))
        user_hts_RT_be_RT_JC = user_hts_RT_be_RT_JC.withColumn('jaccardUsers', array_to_string_udf(col("jc_users")))
        #saves report
        user_hts_RT_be_RT_JC.drop("hts", "rt_users", "bert_users", "jc_users").repartition(1).write.mode("overwrite").csv(
            self.basePath + "\\exportPowerBI", header=True, sep=";")

    def saveFullGraph(self):
        """
        Shortcut for saving full graph

        """
        saveGraph(self.gFull, self.basePath + "\\gFull", "g")

    def extractNeighbours(self, id):
        """
        Extracts subgraph (2 levels of neighbours) starting from id node

        @param id: id of the node's neighbours
        @type id: string
        @return: subgraph (2 levels of neighbours) starting from id node
        @rtype: GraphFrame
        """

        idNeighbours = self.gFull.filterEdges(
            "((src in (" + id + ")) OR (dst in (" + id + "))) AND (type='RT' OR type='HT' OR type='JC')")

        neighbors = [id]
        for row in idNeighbours.edges.collect():
            if row["type"] != "HT":
                neighbors.append(row["dst"])

        neighbors_string = ", ".join(str(item) for item in neighbors)
        gNeighbours = self.gFull.filterEdges(
            "((src in (" + neighbors_string + ")) OR (dst in (" + neighbors_string + "))) AND (type='RT' OR type='HT' OR type='JC')")

        src = gNeighbours.edges.select("src")
        dst = gNeighbours.edges.select("dst")

        return GraphFrame(src.union(dst).distinct().withColumnRenamed("src", "id"), gNeighbours.edges)