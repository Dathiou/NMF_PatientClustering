
package edu.gatech.cse8803.main

import java.text.SimpleDateFormat
import java.util.Random

import edu.gatech.cse8803.clustering
import edu.gatech.cse8803.clustering.{NMF, Metrics}
import edu.gatech.cse8803.features.FeatureConstruction
import edu.gatech.cse8803.ioutils.CSVUtils
import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import edu.gatech.cse8803.phenotyping.T2dmPhenotype
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.clustering.{GaussianMixture, KMeans}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Vectors, Vector}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SchemaRDD, Row, SQLContext}
import org.apache.spark.{rdd, SparkConf, SparkContext}
import java.util.Date
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, sum}
import breeze.linalg._
import scala.io.Source


object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.Logger
    import org.apache.log4j.Level


    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = createContext
    val sqlContext = new SQLContext(sc)

//    /** initialize loading of data */
//
//    val testPhenotypeLabel = sc.parallelize(List(
//      ("pt1",1 ),
//      ("pt2",1 ),
//      ("pt3",2 ),
//      ("pt4",2 ),
//      ("pt5",3 ),
//      ("pt6",3 ),
//      ("pt7",3 )
//    ))
//
//    val testRawFeatures = sc.parallelize(List(
//      ("pt1",Vectors.dense(0.0,0.1)),
//      ("pt2",Vectors.dense(0.1,0.0)),
//      ("pt3",Vectors.dense(3.5,5.5)),
//      ("pt4",Vectors.dense(3.4,5.5)),
//      ("pt5",Vectors.dense(6.0,3.0)),
//      ("pt6",Vectors.dense(6.1,3.0)),
//      ("pt7",Vectors.dense(6.0,3.1))
//    ))
//
//    val (kMeansPurity_test, gaussianMixturePurity_test, nmfPurity_test) = testClustering(testPhenotypeLabel, testRawFeatures)
//    println(f"[All feature] purity of kMeans is: $kMeansPurity_test%.5f")
//    println(f"[All feature] purity of GMM is: $gaussianMixturePurity_test%.5f")
//    println(f"[All feature] purity of NMF is: $nmfPurity_test%.5f")


    val (medication, labResult, diagnostic) = loadRddRawData(sqlContext)
    val (candidateMedication, candidateLab, candidateDiagnostic) = loadLocalRawData





   // test_feature_creation(sc)

    /** feature construction with all features */
    val featureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult),
      FeatureConstruction.constructMedicationFeatureTuple(medication)
    )

   val rawFeatures = FeatureConstruction.construct(sc, featureTuples)
    /** conduct phenotyping */
   val phenotypeLabel = T2dmPhenotype.transform(medication, labResult, diagnostic)
  //  println("unfiltered")
    val (kMeansPurity, gaussianMixturePurity, nmfPurity) = testClustering(phenotypeLabel, rawFeatures)
    println("----------")
    println(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
   println(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
   println(f"[All feature] purity of NMF is: $nmfPurity%.5f")

    /** feature construction with filtered features */
    val filteredFeatureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic, candidateDiagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult, candidateLab),
      FeatureConstruction.constructMedicationFeatureTuple(medication, candidateMedication)
    )

    val filteredRawFeatures = FeatureConstruction.construct(sc, filteredFeatureTuples)
//println("filtered")
    val (kMeansPurity2, gaussianMixturePurity2, nmfPurity2) = testClustering(phenotypeLabel, filteredRawFeatures)
    println("----------")
    println(f"[Filtered feature] purity of kMeans is: $kMeansPurity2%.5f")
    println(f"[Filtered feature] purity of GMM is: $gaussianMixturePurity2%.5f")
    println(f"[Filtered feature] purity of NMF is: $nmfPurity2%.5f")
  }

  def testClustering(phenotypeLabel: RDD[(String, Int)], rawFeatures:RDD[(String, Vector)]): (Double, Double, Double) = {
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix

    /** scale features */
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(rawFeatures.map(_._2))
    val features = rawFeatures.map({ case (patientID, featureVector) => (patientID, scaler.transform(Vectors.dense(featureVector.toArray)))})
    val rawFeatureVectors = features.map(_._2).cache()

    /** reduce dimension */
    val mat: RowMatrix = new RowMatrix(rawFeatureVectors)
   val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
    val featureVectors = mat.multiply(pc).rows.cache()

    val densePc = Matrices.dense(pc.numRows, pc.numCols, pc.toArray).asInstanceOf[DenseMatrix]
    def transform(feature: Vector): Vector = {
     // val scaled = scaler.transform(Vectors.dense(feature.toArray))
      //Vectors.dense(Matrices.dense(1, scaled.size, scaled.toArray).multiply(densePc).toArray)
      Vectors.dense(Matrices.dense(1, feature.size, feature.toArray).multiply(densePc).toArray)
    }


    val clusters = KMeans.train(featureVectors, 3, 20, 1, "k-means||", 0L)
    val kmeans_pred = clusters.predict(featureVectors)
    val assignmentsKmeans=features.map({case (patientId,f)=>patientId}).zip(kmeans_pred)
    val KmeansAssignmentAndLabel = assignmentsKmeans.join(phenotypeLabel).map({case (patientID,value)=>value})
    val kMeansPurity = Metrics.purity(KmeansAssignmentAndLabel)

    val gmmm = new GaussianMixture().setK(3).setMaxIterations(20).setSeed(0L).run(featureVectors)
    val gm_pred = gmmm.predict(featureVectors);
    val assignmentsGMM=features.map({case (patientId,f)=>patientId}).zip(gm_pred)
    val GMMAssignmentAndLabel = assignmentsGMM.join(phenotypeLabel).map({case (patientID,value)=>value})
    val gaussianMixturePurity = Metrics.purity(GMMAssignmentAndLabel)
//for (k <- 2 to 5)
//    {
      /** to NMF */
      val (w, _) = NMF.run(new RowMatrix(rawFeatureVectors), 3, 100)
      // for each row (patient) in W matrix, the index with the max value should be assigned as its cluster type
      val assignments = w.rows.map(_.toArray.zipWithIndex.maxBy(_._1)._2)
      // zip patientIDs with their corresponding cluster assignments
      // Note that map doesn't change the order of rows
      val assignmentsWithPatientIds = features.map({ case (patientId, f) => patientId }).zip(assignments)
      // join your cluster assignments and phenotypeLabel on the patientID and obtain a RDD[(Int,Int)]
      // which is a RDD of (clusterNumber, phenotypeLabel) pairs
      val nmfClusterAssignmentAndLabel = assignmentsWithPatientIds.join(phenotypeLabel).map({ case (patientID, value) => value })

   //val o = Metrics.testClustering(nmfClusterAssignmentAndLabel)
      // Obtain purity value
      val nmfPurity = Metrics.purity(nmfClusterAssignmentAndLabel)

      //print("nmf k="+k.toString+": "+nmfPurity.toString)

 //   }

   (kMeansPurity, gaussianMixturePurity, nmfPurity)

  }

  /**
   * load the sets of string for filtering of medication
   * lab result and diagnostics
    *
    * @return
   */
  def loadLocalRawData: (Set[String], Set[String], Set[String]) = {
    val candidateMedication = Source.fromFile("data/med_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateLab = Source.fromFile("data/lab_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateDiagnostic = Source.fromFile("data/icd9_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    (candidateMedication, candidateLab, candidateDiagnostic)
  }


  def loadRddRawData(sqlContext: SQLContext): (RDD[Medication], RDD[LabResult], RDD[Diagnostic]) = {
    /** You may need to use this date format. */
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")

    /** load data using Spark SQL into three RDDs and return them
      * Hint: You can utilize: edu.gatech.cse8803.ioutils.CSVUtils and SQLContext
      *       Refer to model/models.scala for the shape of Medication, LabResult, Diagnostic data type
      *       Be careful when you deal with String and numbers in String type
      * */

    /** TODO: implement your own code here and remove existing placeholder code below */

    val x  = CSVUtils.loadCSVAsTable(sqlContext: SQLContext, "data/lab_results_INPUT.csv": String , "LAB")
    val y  = CSVUtils.loadCSVAsTable(sqlContext: SQLContext, "data/medication_orders_INPUT.csv": String , "MED")
    val z1  = CSVUtils.loadCSVAsTable(sqlContext: SQLContext, "data/encounter_INPUT.csv": String , "DIAG")
    val z2  = CSVUtils.loadCSVAsTable(sqlContext: SQLContext, "data/encounter_dx_INPUT.csv": String , "DIAGDX")
    //data.printSchema()

    val lab = sqlContext.sql("SELECT Member_ID as patientID, Date_Collected as date,Result_Name as testName,Numeric_Result as value FROM LAB WHERE Numeric_Result!=0 OR Numeric_Result='200,000'")
    val lab_rdd : RDD[LabResult] = lab.rdd.map {r: Row => new LabResult(r.getString(0),dateFormat.parse(r.getString(1)),r.getString(2).toLowerCase,java.lang.Double.valueOf(r.getString(3).replace(",", "")))}
    //lab.printSchema()
    val med = sqlContext.sql("SELECT Member_ID as patientID, Order_Date as date,Drug_Name as medicine FROM MED")
    val med_rdd : RDD[Medication] = med.rdd.map {r: Row => new Medication(r.getString(0),dateFormat.parse(r.getString(1)),r.getString(2).toLowerCase)}
    //med_rdd.take(5).foreach(print)
   // lab_rdd.take(5).foreach(print)
    //ab_rdd.
    val diag = sqlContext.sql("SELECT Member_ID as patientID, Encounter_DateTime as date,DIAGDX.code as code FROM DIAG JOIN DIAGDX ON DIAG.Encounter_ID=DIAGDX.Encounter_ID")
    val diag_rdd : RDD[Diagnostic] = diag.rdd.map {r: Row => new Diagnostic(r.getString(0),dateFormat.parse(r.getString(1)),r.getString(2).toLowerCase)}

    val medication: RDD[Medication] =   med_rdd
    val labResult: RDD[LabResult] =  lab_rdd
    val diagnostic: RDD[Diagnostic] =  diag_rdd
    (medication, labResult, diagnostic)
  }

  def createContext(appName: String, masterUrl: String): SparkContext = {
    val conf = new SparkConf().setAppName(appName).setMaster(masterUrl)
    new SparkContext(conf)
  }

  def createContext(appName: String): SparkContext = createContext(appName, "local")

  def createContext: SparkContext = createContext("CSE 8803 Homework Two Application", "local")

  object data

  object lab



}
