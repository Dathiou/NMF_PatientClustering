package edu.gatech.cse8803.clustering


import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, sum}
import breeze.linalg._
import breeze.numerics._
import edu.gatech.cse8803.clustering
import org.apache.spark.mllib.linalg.{Matrices, Vectors, Vector}
import org.apache.spark.mllib.linalg.distributed.{DistributedMatrix, RowMatrix}
import org.apache.spark.rdd.RDD


object NMF {

  /**
   * Run NMF clustering
 *
   * @param V The original non-negative matrix 
   * @param k The number of clusters to be formed, also the number of cols in W and number of rows in H
   * @param maxIterations The maximum number of iterations to perform
   * @param convergenceTol The maximum change in error at which convergence occurs.
   * @return two matrixes W and H in RowMatrix and DenseMatrix format respectively 
   */
  def run(V: RowMatrix, k: Int, maxIterations: Int, convergenceTol: Double = 1e-4): (RowMatrix, BDM[Double]) = {

    /**
     * TODO 1: Implement your code here
     * Initialize W, H randomly 
     * Calculate the initial error (Euclidean distance between V and W * H)
     */
   val r = k

    var W = new RowMatrix(V.rows.map(_ => BDV.rand[Double](r)).map(fromBreeze).cache)

    var H = BDM.rand[Double](r,V.numCols().toInt)
    //print(H)
    var eucl = Dist(V,W,H)


    //.map(case (r1, r2) => r1-r2)//.map((r1, r2) => r1-r2)
    //val A = V - multiply(W ,H);

 var i =0


    V.rows.cache()
    /**
     * TODO 2: Implement your code here
     * Iteratively update W, H in a parallel fashion until error falls below the tolerance value 
     * The updating equations are, 
     * H = H.* W^T^V ./ (W^T^W H)
     * W = W.* VHste^T^ ./ (W H H^T^)
    */

    while(eucl > convergenceTol && i<maxIterations) {

      var H_s = H * H.t
      var VHT = multiply(V, H.t)
      var WHHT = multiply(W, H_s)
      W = dotProd(W, dotDiv(VHT, WHHT))

      W.rows.cache()
      var WTV = computeWTV(W, V)
      var W_s = computeWTV(W, W)
      var WTWH = W_s * H

      H = (H :* WTV) :/ (WTWH :+ 0.0001)

      var eucl= Dist(V, W, H)

     // print(" iteration: "+ i.toString + " - error: "+eucl.toString + "\r\n")
      i= i+1
    }


    /** TODO: Remove the placeholder for return and replace with correct values */
    (W,H)
  }


  /**  
  * TODO: Implement the helper functions if you needed
  * Below are recommended helper functions for matrix manipulation
  * For the implementation of the first three helper functions (with a null return), 
  * you can refer to dotProd and dotDiv whose implementation are provided
  */
  /**
  * Note:You can find some helper functions to convert vectors and matrices
  * from breeze library to mllib library and vice versa in package.scala
  */

  /** compute the mutiplication of a RowMatrix and a dense matrix */
   def multiply(X: RowMatrix, d: BDM[Double]): RowMatrix = {


 X.multiply(fromBreeze(d))

  }

 /** get the dense matrix representation for a RowMatrix */
  def getDenseMatrix(X: RowMatrix): BDM[Double] = {

   val A = new DenseMatrix(X.numCols().toInt,X.numRows().toInt,X.rows.collect.flatMap(x => x.toArray))
A.t
    }

  /** matrix multiplication of W.t and V */
  def computeWTV(W: RowMatrix, V: RowMatrix): BDM[Double] = {

    //getDenseMatrix(multiply(W, getDenseMatrix(V)))
    //getDenseMatrix(W).t*getDenseMatrix(V)
    W.rows.zip(V.rows).map{
      case(w: Vector, v: Vector) => BDM.create(w.size, 1, w.toArray) * BDM.create(1, v.size, v.toArray)
    }.reduce(_+_)

//w.getDenseMatrix *

  }



  /** sub of two RowMatrixes */
  def Dist(V: RowMatrix, W: RowMatrix,H: BDM[Double]): Double = {
    val A = multiply(W ,H)
    val X = Sub(V,A)
    dotProd(X,X).rows.map(a => a.toArray.sum).sum()/2//.reduce{_:+_}//.toArray.reduce{_+_}

  }


  /** sub of two RowMatrixes */
  def Sub(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :- toBreezeVector(v2)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  /** dot product of two RowMatrixes */
  def Add(X: RowMatrix, Y: Double): RowMatrix = {
    val rows = X.rows.map{v1: Vector =>
      toBreezeVector(v1) :+ Y
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  /** dot product of two RowMatrixes */
  def dotProd(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :* toBreezeVector(v2)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  /** dot division of two RowMatrixes */
  def dotDiv(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :/ toBreezeVector(v2).mapValues(_ + 2.0e-15)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }
}