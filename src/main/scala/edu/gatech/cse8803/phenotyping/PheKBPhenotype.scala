/**
  * @author Hang Su <hangsu@gatech.edu>,
  * @author Sungtae An <stan84@gatech.edu>,
  */

package edu.gatech.cse8803.phenotyping

import breeze.linalg.min
import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import java.util.Date



object T2dmPhenotype {
  /**
    * Transform given data set to a RDD of patients and corresponding phenotype
    *
    * @param medication medication RDD
    * @param labResult lab result RDD
    * @param diagnostic diagnostic code RDD
    * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
    */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {
    /**
      * Remove the place holder and implement your code here.
      * Hard code the medication, lab, icd code etc for phenotype like example code below
      * as while testing your code we expect your function have no side effect.
      * i.e. Do NOT read from file or write file
      *
      * You don't need to follow the example placeholder codes below exactly, once you return the same type of return.
      */

    val sc = medication.sparkContext
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val tot = sc.union(diagnostic.map(_.patientID),medication.map(_.patientID),labResult.map(_.patientID))


    /** Hard code the criteria */
    val type1_dm_dx = Set("code1","250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43", "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93")
    val type2_dm_dx = Set("code2","250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6", "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02")

    val type1_dm_med = Set("med1", "lantus", " insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente","insulin,ultralente")
    val type2_dm_med = Set("med2", "chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl", "glucatrol", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl", "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose", "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide", "avandia", "actos", "ACTOS","glipizide")

    /** Find CASE Patients */
    val dm1_diag = diagnostic.filter{di => type1_dm_dx(di.code)}.map(_.patientID)
    val dm2_diag = diagnostic.filter{di => type2_dm_dx(di.code)}.map(_.patientID)
    val dm1_med = medication.filter {di => type1_dm_med(di.medicine)}.map(_.patientID)
    val dm2_med = medication.filter {di => type2_dm_med(di.medicine)}.map(_.patientID)

    val step2 =tot.subtract(dm1_diag).intersection(dm2_diag)
    val step3= step2.intersection (dm1_med)
    val case1 = step2.subtract(step3)

    val step4=step3.intersection(dm2_med)
    val case2=step3.subtract(step4)


    val date2 = medication.filter( a => type2_dm_med(a.medicine)).map(a=> (a.patientID,a.date.getTime)).reduceByKey((x,y) => math.min(x,y))
    val date1 = medication.filter( a => type1_dm_med(a.medicine)).map(a=> (a.patientID,a.date.getTime)).reduceByKey((x,y) => math.min(x,y))
    val step4prep = date1.join(date2)

    val step5 = step4prep.filter(a => a._2._1 >= a._2._2).map(_._1)

    val case3 = step4.intersection(step5)

    val casePatients = sc.union(case1,case2,case3).distinct()



    /** Find CONTROL Patients */
    val glu = labResult.filter{a => a.testName.contains("glucose")}.map(_.patientID).distinct()

    val ab = labResult.filter{ a =>
      val name = a.testName
      val value = a.value
      ((name=="hba1c" && value>=6.0) || (name=="hemoglobin a1c" && value>=6.0) || (name=="fasting glucose" && value>=110)|| (name=="fasting blood glucose" && value>=110)|| (name=="fasting plasma glucose" && value>=110)|| (name=="glucose" && value>110)|| (name=="glucose, serum" && value>110))
    }.map{_.patientID}

    val mellitus = Set("790.21", "790.22", "790.2", "790.29", "648.81", "648.82", "648.83", "648.84", "648", "648", "648.01", "648.02", "648.03", "648.04", "791.5", "277.7", "V77.1", "256.4")

    val MD = diagnostic.filter{ a => (mellitus(a.code) || a.code.take(3) == "250") }.map(_.patientID)

    val controlPatients = tot.intersection(glu).subtract(ab).subtract(MD).distinct()

    /** Find OTHER Patients */
    val others = tot.subtract(casePatients).subtract(controlPatients).distinct()


    /** Once you find patients for each group, make them as a single RDD[(String, Int)] */
    val phenotypeLabel = sc.union(casePatients.map(a=> (a,1)),controlPatients.map(a=> (a,2)),others.map(a=> (a,3)))


    phenotypeLabel
    //casePatients
  }
}
