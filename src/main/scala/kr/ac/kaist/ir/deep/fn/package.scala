package kr.ac.kaist.ir.deep

import breeze.linalg.DenseMatrix
import play.api.libs.json.{JsArray, JsNumber}

/**
 * Package for various functions.
 */
package object fn {
  /** Type of scalar **/
  type Scalar = Double
  /** Type of probability **/
  type Probability = Double
  /** Type of Neuron Input **/
  type ScalarMatrix = DenseMatrix[Scalar]
  /** Define Alias **/
  val Tanh = HyperbolicTangent

  /**
   * Defines sugar operations for ScalarMatrix
   *
   * @param x the __matrix__ to be computed
   */
  implicit class ScalarMatrixOp(x: ScalarMatrix) {
    /**
     * Add __given scalar__ to last row.
     *
     * @param y a __scalar__ to be added
     */
    def row_+(y: Scalar): ScalarMatrix = {
      val scalar: ScalarMatrix = (ScalarMatrix $1(1, x.cols)) :* y
      x row_+ scalar
    }

    /**
     * Add __given matrix__ to last rows.
     *
     * @param y a __matrix__ to be added
     */
    def row_+(y: ScalarMatrix): ScalarMatrix = {
      DenseMatrix.vertcat(x, y)
    }

    /**
     * Add __given matrix__ to last columns.
     *
     * @param y a __matrix__ to be added
     */
    def col_+(y: ScalarMatrix) = {
      DenseMatrix.horzcat(x, y)
    }

    /**
     * Make given matrix as 2D JSON Array
     *
     * @return JsArray of this matrix
     */
    def to2DSeq: JsArray = {
      val r = x.rows
      val c = x.cols
      JsArray((0 until r) map {
        i ⇒ JsArray((0 until c) map {
          j ⇒ JsNumber(x(i, j))
        })
      })
    }

    /**
     * String representation of matrix
     *
     * @return string representation
     */
    def mkString: String =
      "{" + (((0 until x.rows) map {
        r ⇒ "[" + (((0 until x.cols) map { c ⇒ f"${x(r, c)}%.3f"}) mkString ", ") + "]"
      }) mkString ", ") + "}"
  }

  /**
   * Defines sugar operations of probability
   *
   * @param x __scalar__ to be applied
   */
  implicit class ProbabilityOp(x: Probability) {
    /**
     * Returns safe probability
     *
     * @return probability between 0 and 1
     */
    def safe = if (0.0 <= x && x <= 1.0) x else if (x < 0.0) 0.0 else 1.0
  }

  /**
   * Defines sugar operations of sequence of weights
   *
   * @param w __matrix sequence__ to be applied.
   */
  implicit class WeightSeqOp(w: IndexedSeq[ScalarMatrix]) {
    /**
     * Assign scalar 
     *
     * @param x __scalar__ to be assigned for every cell
     */
    def :=(x: Scalar) = w.par foreach {
      _ := x
    }

    /**
     * Assign matrices 
     * @param w2 to be assigned
     */
    def :=(w2: IndexedSeq[ScalarMatrix]) = w.indices.par foreach { id ⇒ w(id) := w2(id)}

    /**
     * Add onto another matrices if they are exists, otherwise copy these matrices.
     *
     * @param w2 __matrix sequence__ to be added onto.
     * @return added matrices
     */
    def copy_+(w2: IndexedSeq[ScalarMatrix]) =
      if (w2.isEmpty) copy
      else {
        w.indices.par foreach {
          id ⇒ w2(id) :+= w(id)
        }
        w2
      }

    /**
     * Copy these matrices
     *
     * @return copied matrices
     */
    def copy = w map {
      _.copy
    }

    /**
     * Add another matrices in-place. 
     *
     * @param w2 __matrix sequence__ to be added
     */
    def :+=(w2: IndexedSeq[ScalarMatrix]) = {
      w.indices.par foreach { id ⇒ w(id) :+= w2(id)}
      w
    }

    /**
     * Divide matrices with given scalar
     *
     * @param x __scalar__ as a divider.
     */
    def :/=(x: Scalar) = {
      w foreach { matx ⇒ matx :/= x}
    }
  }

}
