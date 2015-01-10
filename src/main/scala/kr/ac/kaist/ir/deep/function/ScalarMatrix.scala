package kr.ac.kaist.ir.deep.function

import breeze.linalg.DenseMatrix

/**
 * Companion Object of ScalarMatrix
 */
object ScalarMatrix {
  /**
   * Generates full-one matrix of given size
   * @param size of matrix, such as (2, 3)
   * @return Matrix with initialized by one
   */
  def $1(size: (Int, Int)) = DenseMatrix.ones[Scalar](size._1, size._2)

  /**
   * Generates full-random matrix of given size
   * @param size of matrix, such as (2, 3)
   * @return Matrix with initialized by random number
   */
  def of(size: (Int, Int)) = DenseMatrix.tabulate[Scalar](size._1, size._2)((_, _) ⇒ Math.random())

  /**
   * Generate full 0-1 matrix of given size. Probability of 1 is given.
   * @param pair is pair of (row, col, probability)
   * @return generated matrix
   */
  def $01(pair: (Int, Int, Probability)) = DenseMatrix.tabulate[Scalar](pair._1, pair._2)((_, _) ⇒ if (Math.random() > pair._3) 0.0 else 1.0)

  /**
   * Restore a matrix from JSON seq.
   * @param arr to be restored
   * @return restored matrix
   */
  def restore(arr: Seq[Seq[Scalar]]) = {
    val res = $0(arr.size, arr(0).size)
    arr.indices.par foreach {
      r ⇒ arr(r).indices.par foreach {
        c ⇒ res.update(r, c, arr(r)(c))
      }
    }
    res
  }

  /**
   * Generates full-zero matrix of given size
   * @param size of matrix, such as (2, 3)
   * @return Matrix with initialized by zero
   */
  def $0(size: (Int, Int)) = DenseMatrix.zeros[Scalar](size._1, size._2)
}

