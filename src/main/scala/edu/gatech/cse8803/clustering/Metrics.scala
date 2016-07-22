/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse8803.clustering

import breeze.linalg.{max, sum}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

object Metrics {
  /**
   * Given input RDD with tuples of assigned cluster id by clustering,
   * and corresponding real class. Calculate the purity of clustering.
   * Purity is defined as
   *             \fract{1}{N}\sum_K max_j |w_k \cap c_j|
   * where N is the number of samples, K is number of clusters and j
   * is index of class. w_k denotes the set of samples in k-th cluster
   * and c_j denotes set of samples of class j.
    *
    * @param clusterAssignmentAndLabel RDD in the tuple format
   *                                  (assigned_cluster_id, class)
   * @return purity
   */
  def purity(clusterAssignmentAndLabel: RDD[(Int, Int)]): Double = {
    /**
     * TODO: Remove the placeholder and implement your code here
     */

    val sumAffectedDatas = clusterAssignmentAndLabel.map(d => (d, 1))
      .reduceByKey{_+_}

    val maxByCluster = sumAffectedDatas.map(sa => (sa._1._1, sa._2))
      .reduceByKey{case (sum1, sum2) => sum1.max(sum2) }
      .map(_._2)
      .collect()

    maxByCluster.sum / clusterAssignmentAndLabel.count().toDouble
  }

  def testClustering(pred: RDD[(Int, Int)]) ={
    val r = pred.groupBy(_._2).map {case (a,b) =>

      val i = b.toSeq.groupBy(_._1).map { c =>
      val t = c._2.size.toDouble/b.size.toDouble
        (c._1,t)
      }

      (a,i)
    }

  // val r = pred.groupBy(_._2)
    println("test")
    r.take(5).foreach(println)


  }






}
