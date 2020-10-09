import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession

from Rolling_avg import Rolling_average


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    database = "baseball"
    port = "3306"
    user = " "
    password = " "

    baseball = (
        spark.read.format("jdbc")
        .options(
            url=f"jdbc:mysql://localhost:{port}/{database}",
            driver="com.mysql.cj.jdbc.Driver",
            dbtable="(select Hit,Atbat,batter,b.game_id,DATE(g.local_date) as local_dt \
                FROM batter_counts b join game g on g.game_id=b.game_id)batters",
            user=user,
            password=password,
        )
        .load()
    )

    # Loading the table in a Temp view so that we can reuse it
    baseball.createOrReplaceTempView("batters")
    baseball.persist(StorageLevel.DISK_ONLY)

    # Simple SQL
    batter = spark.sql("(SELECT * FROM batters)")
    batter.show()

    # Rolling average
    results = spark.sql(
        """(SELECT SUM(ba.Hit) as Hits, SUM(ba.atbat) as AtBat, b.game_id,b.batter \
                           FROM batters b JOIN \
                           batters ba ON ba.batter = b.batter AND \
                           ba.local_dt BETWEEN DATE_SUB(b.local_dt,100) AND \
                           DATE_SUB(b.local_dt, 1)\
                           GROUP BY b.game_id, b.batter \
                           ORDER BY b.game_id)"""
    )

    # Let's calculate the Rolling average using the Transformer
    rolling_average = Rolling_average(
        inputCols=["Hits", "AtBat"], outputCol="Batter_Rolling_avg"
    )
    bat_rolling_avg = rolling_average.transform(results)
    print("100 Day rolling average\n")
    bat_rolling_avg.show()


if __name__ == "__main__":
    sys.exit(main())
