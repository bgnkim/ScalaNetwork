package kr.ac.kaist.ir.deep.fn

import breeze.linalg.DenseMatrix

/**
 * Companion Object of ScalarMatrix
 *
 * This object defines various shortcuts. 
 */
object ScalarMatrix {
  /**
   * Generates full-one matrix of given size
   *
   * @param size __(#row, #col) pair__ of matrix size, such as (2, 3)
   * @return Matrix with initialized by one
   */
  def $1(size: (Int, Int)): ScalarMatrix = DenseMatrix.ones[Scalar](size._1, size._2)

  /**
   * Generates full-random matrix of given size
   *
   * @param size __(#row, #col) pair__ of matrix size, such as (2, 3)
   * @return Matrix with initialized by random number
   */
  def of(size: (Int, Int)): ScalarMatrix =
    DenseMatrix.tabulate[Scalar](size._1, size._2)((_, _) ⇒ Math.random())

  /**
   * Generate full 0-1 matrix of given size. __Probability of 1's occurrence__ is given.
   *
   * @param pair __(#row, #col, probability)__ pair, where (#row, #col) indicates the matrix size, probability indicates the probability of 1's occurrence.
   * @return generated matrix
   */
  def $01(pair: (Int, Int, Probability)): ScalarMatrix =
    DenseMatrix.tabulate[Scalar](pair._1, pair._2)((_, _) ⇒ if (Math.random() > pair._3) 0.0 else 1.0)

  /**
   * Restore a matrix from JSON seq.
   *
   * @param arr 2D Sequence to be restored
   * @return restored matrix
   */
  def restore(arr: Seq[Seq[Scalar]]): ScalarMatrix = {
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
   *
   * @param size __(#row, #col) pair__ of matrix size, such as (2, 3)
   * @return Matrix with initialized by zero
   */
  def $0(size: (Int, Int)): ScalarMatrix = DenseMatrix.zeros[Scalar](size._1, size._2)

  /**
   * Make a column vector with given sequence.
   *
   * @param seq Sequence of entries, from (1,1) to (size, 1).
   * @return column vector with given sequence
   */
  def apply(seq: Double*): ScalarMatrix = DenseMatrix.create(seq.size, 1, seq.toArray)
}

