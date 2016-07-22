/**
 * @author Hang Su
 */
package edu.gatech.cse8803.features

import edu.gatech.cse8803.model.{LabResult, Medication, Diagnostic}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._


object FeatureConstruction {

  /**
   * ((patient-id, feature-name), feature-value)
   */
  type FeatureTuple = ((String, String), Double)

  /**
   * Aggregate feature tuples from diagnostic with COUNT aggregation,
    *
    * @param diagnostic RDD of diagnostic
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]): RDD[FeatureTuple] = {
    diagnostic.map(a => ((a.patientID,a.code),1.0)).reduceByKey(_+_)//.groupBy(_._1).mapValues(_.map(_._2.code).size.toDouble)

  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation,
    *
    * @param medication RDD of medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication]): RDD[FeatureTuple] = {
medication.map(a => ((a.patientID,a.medicine),1.0)).reduceByKey(_+_)// .groupBy(_._1).mapValues(_.map(_._2.medicine).size.toDouble)
  }

  /**
   * Aggregate feature tuples from lab result, using AVERAGE aggregation
    *
    * @param labResult RDD of lab result
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult]): RDD[FeatureTuple] = {
labResult.map(a => ((a.patientID,a.testName),a)).groupBy(_._1).map{ case (a,b) => (a,b.map(_._2.value).sum / b.size)}
  }

  /**
   * Aggregate feature tuple from diagnostics with COUNT aggregation, but use code that is
   * available in the given set only and drop all others.
    *
    * @param diagnostic RDD of diagnostics
   * @param candidateCode set of candidate code, filter diagnostics based on this set
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic], candidateCode: Set[String]): RDD[FeatureTuple] = {
    diagnostic.filter{a=> candidateCode(a.code)}.map(a => ((a.patientID,a.code),a)).groupBy(_._1).mapValues(_.map(_._2.code).size.toDouble)
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation, use medications from
   * given set only and drop all others.
    *
    * @param medication RDD of diagnostics
   * @param candidateMedication set of candidate medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication], candidateMedication: Set[String]): RDD[FeatureTuple] = {
    medication.filter{a => candidateMedication(a.medicine)}.map(a => ((a.patientID,a.medicine),a)).groupBy(_._1).mapValues(_.map(_._2.medicine).size.toDouble)
  }


  /**
   * Aggregate feature tuples from lab result with AVERAGE aggregation, use lab from
   * given set of lab test names only and drop all others.
    *
    * @param labResult RDD of lab result
   * @param candidateLab set of candidate lab test name
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult], candidateLab: Set[String]): RDD[FeatureTuple] = {
labResult.filter{a => candidateLab(a.testName)}.map(a => ((a.patientID,a.testName),a)).groupBy(_._1).map{ case (a,b) => (a,b.map(_._2.value).sum / b.size)}
  }


  /**
   * Given a feature tuples RDD, construct features in vector
   * format for each patient. feature name should be mapped
   * to some index and convert to sparse feature format.
    *
    * @param sc SparkContext to run
   * @param feature RDD of input feature tuples
   * @return
   */
  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()

    /** create a feature name to id map*/
    val id = feature.groupBy(_._1._2).zipWithIndex.map(a => (a._1._1,a._2.toInt))
    val si = id.collect.size

    /** transform input feature */

    val feat_transformed = feature.map(a => (a._1._2,a)).join(id).map(a=> (a._2._1._1._1,(a._2._2,a._2._1._2)))
    val features_final = feat_transformed.groupByKey()
      val f2=  features_final.map{case (tar,feat) =>
        val featvec = Vectors.sparse(si,feat.toSeq)
        (tar,featvec)
    }


    f2
    /** The feature vectors returned can be sparse or dense. It is advisable to use sparse */
  }
}


