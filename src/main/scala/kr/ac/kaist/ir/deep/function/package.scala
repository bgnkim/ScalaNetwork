package kr.ac.kaist.ir.deep

import breeze.linalg._
import play.api.libs.json.{JsArray, JsNumber}

/**
 * Package for functions
 *
 * Created by bydelta on 2014-12-27.
 */
package object function {
  /** Type of scalar **/
  type Scalar = Double
  /** Type of probability **/
  type Probability = Double
  /** Type of Neuron Input **/
  type ScalarMatrix = DenseMatrix[Scalar]
  /** Define Alias **/
  val Tanh = HyperbolicTangent

  /**
   * Define Activation Function trait.
   */
  trait Activation extends (ScalarMatrix ⇒ ScalarMatrix) with Serializable {
    /**
     * Compute derivative of this function
     * @param fx is output of this function
     * @return differentiation value at f(x)=fx, which is Square, diagonal matrix
     */
    def derivative(fx: ScalarMatrix): ScalarMatrix

    /**
     * Compute mapping at x
     * @param x is input scalar.
     * @return f(x)
     */
    def apply(x: ScalarMatrix): ScalarMatrix

    /**
     * Initialize Weight matrix
     * @param fanIn is a weight vector indicates fan-in
     * @param fanOut is a count of fan-out
     * @return weight matrix
     */
    def initialize(fanIn: Int, fanOut: Int, rows: Int = 0, cols: Int = 0): ScalarMatrix = {
      val pmMatx: ScalarMatrix = ScalarMatrix.of(if (rows > 0) rows else fanOut, if (cols > 0) cols else fanIn) :* 1e-2
      pmMatx :+ 1e-2
    }
  }

  /**
   * Define Objective Function trait.
   */
  trait Objective extends ((ScalarMatrix, ScalarMatrix) ⇒ Scalar) with Serializable {
    /**
     * Compute derivative of this objective function
     * @param real is expected real output
     * @param output is computational output of the network
     * @return is differentiation(Gradient) vector at f(X)=output, i.e. error of each output neuron.
     */
    def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix

    /**
     * Compute error
     * @param real is expected real output
     * @param output is computational output of the network
     * @return is the error
     */
    override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar
  }

  /**
   * Defines sugar operations for ScalarMatrix
   * @param x to be computed
   */
  implicit class ScalarMatrixOp(x: ScalarMatrix) {
    /**
     * Add given scalar to last row.
     * @param y to be added
     */
    def row_+(y: Scalar): ScalarMatrix = {
      val scalar: ScalarMatrix = (ScalarMatrix $1(1, x.cols)) :* y
      x row_+ scalar
    }

    /**
     * Add given matrix to last rows.
     * @param y to be added
     */
    def row_+(y: ScalarMatrix): ScalarMatrix = {
      DenseMatrix.vertcat(x, y)
    }
    
    /**
     * Add given matrix to last columns.
     * @param y to be added
     */
    def col_+(y: ScalarMatrix) = {
      DenseMatrix.horzcat(x, y)
    }

    /**
     * Make 2D Sequence
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

    def mkString: String =
      "{" + (((0 until x.rows) map {
        r ⇒ "[" + (((0 until x.cols) map { c ⇒ f"${x(r, c)}%.3f"}) mkString ", ") + "]"
      }) mkString ", ") + "}"
  }

  /**
   * Defines sugar operations of probability
   * @param x to be applied
   */
  implicit class ProbabilityOp(x: Probability) {
    /**
     * Returns safe probability
     * @return probability between 0 and 1
     */
    def safe = if (0.0 <= x && x <= 1.0) x else if (x < 0.0) 0.0 else 1.0
  }

  /**
   * Defines sugar operations of sequence of weights 
   * @param w to be applied.
   */
  implicit class WeightSeqOp(w: Seq[ScalarMatrix]) {
    /**
     * Assign scalar 
     * @param x to be assigned
     */
    def :=(x: Scalar) = w.par foreach {
      _ := x
    }

    /**
     * Assign matrices 
     * @param w2 to be assigned
     */
    def :=(w2: Seq[ScalarMatrix]) = w.indices.par foreach { id ⇒ w(id) := w2(id)}

    /**
     * Add onto another matrices if they are exists, otherwise copy these matrices.
     * @param w2 to be added onto.
     * @return added matrices
     */
    def copy_+(w2: Seq[ScalarMatrix]) =
      if (w2.isEmpty) copy
      else {
        w.indices.par foreach {
          id ⇒ w2(id) :+= w(id)
        }
        w2
      }

    /**
     * Copy these matrices
     * @return copied matrices
     */
    def copy = w map {
      _.copy
    }

    /**
     * Add another matrices in-place. 
     * @param w2 to be added
     */
    def :+=(w2: Seq[ScalarMatrix]) = {
      w.indices.par foreach { id ⇒ w(id) :+= w2(id)}
      w
    }
  }
}
