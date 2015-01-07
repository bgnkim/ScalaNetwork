package kr.ac.kaist.ir.deep

import breeze.linalg._
import breeze.numerics.{exp, log, sigmoid, tanh}
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
    def row_+(y: Scalar) = {
      val scalar: ScalarMatrix = (ScalarMatrix $1(1, x.cols)) :* y
      DenseMatrix.vertcat(x, scalar)
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
      arr.indices foreach {
        r ⇒ arr(r).indices foreach {
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

  /**
   * Activation Function: Sigmoid
   */
  object Sigmoid extends Activation {
    /**
     * Compute derivative of this function
     * @param fx is output of this function
     * @return differentiation value at f(x)=fx, which is Square, diagonal matrix
     */
    override def derivative(fx: ScalarMatrix): ScalarMatrix = {
      // Because fx is n by 1 matrix, generate n by n matrix
      val res = ScalarMatrix $0(fx.rows, fx.rows)
      // Output is diagonal matrix, with dfi(xi)/dxi.
      (0 until fx.rows) foreach { r ⇒ {
        val x = fx(r, 0)
        res.update((r, r), x * (1.0 - x))
      }
      }
      res
    }

    /**
     * Compute mapping at x
     * @param x is input scalar.
     * @return f(x)
     */
    override def apply(x: ScalarMatrix): ScalarMatrix = sigmoid(x)

    /**
     * Initialize Weight matrix
     * @param fanIn is a weight vector indicates fan-in
     * @param fanOut is a count of fan-out
     * @return weight matrix
     */
    override def initialize(fanIn: Int, fanOut: Int, rows: Int = 0, cols: Int = 0): ScalarMatrix = {
      val range = Math.sqrt(6.0 / (fanIn + fanOut)) * 4.0
      val pmMatx: ScalarMatrix = ScalarMatrix.of(if (rows > 0) rows else fanOut, if (cols > 0) cols else fanIn) :- 0.5
      pmMatx :* (2.0 * range)
    }
  }

  /**
   * Activation Function: Tanh
   */
  object HyperbolicTangent extends Activation {
    /**
     * Compute derivative of this function
     * @param fx is output of this function
     * @return differentiation value at f(x)=fx, which is Square, diagonal matrix
     */
    override def derivative(fx: ScalarMatrix): ScalarMatrix = {
      // Because fx is n by 1 matrix, generate n by n matrix
      val res = ScalarMatrix $0(fx.rows, fx.rows)
      // Output is diagonal matrix, with dfi(xi)/dxi.
      (0 until fx.rows) foreach { r ⇒ {
        val x = fx(r, 0)
        res.update((r, r), 1.0 - x * x)
      }
      }
      res
    }

    /**
     * Compute mapping at x
     * @param x is input scalar.
     * @return f(x)
     */
    override def apply(x: ScalarMatrix): ScalarMatrix = tanh(x)

    /**
     * Initialize Weight matrix
     * @param fanIn is a weight vector indicates fan-in
     * @param fanOut is a count of fan-out
     * @return weight matrix
     */
    override def initialize(fanIn: Int, fanOut: Int, rows: Int = 0, cols: Int = 0): ScalarMatrix = {
      val range = Math.sqrt(6.0 / (fanIn + fanOut))
      val pmMatx: ScalarMatrix = ScalarMatrix.of(if (rows > 0) rows else fanOut, if (cols > 0) cols else fanIn) :- 0.5
      pmMatx :* (2.0 * range)
    }
  }

  /**
   * Activation Function: Rectifier
   *
   * Rectifier(x) = x if x > 0, otherwise 0
   */
  object Rectifier extends Activation {
    /**
     * Compute derivative of this function
     * @param fx is output of this function
     * @return differentiation value at f(x)=fx, which is Square, diagonal matrix
     */
    override def derivative(fx: ScalarMatrix): ScalarMatrix = {
      // Because fx is n by 1 matrix, generate n by n matrix
      val res = ScalarMatrix $0(fx.rows, fx.rows)
      // Output is diagonal matrix, with dfi(xi)/dxi.
      (0 until fx.rows) foreach { r ⇒ {
        val x = fx(r, 0)
        res.update((r, r), if (x > 0) 1.0 else 0.0)
      }
      }
      res
    }

    /**
     * Compute mapping at x
     * @param x is input scalar.
     * @return f(x)
     */
    override def apply(x: ScalarMatrix): ScalarMatrix = {
      val res = x.copy
      x foreachKey { key ⇒ if (x(key) < 0) res.update(key, 0.0)}
      res
    }
  }

  /**
   * Activation Function: Softplus
   *
   * Softplus(x) = log(1 + e ** x)
   */
  object Softplus extends Activation {
    /**
     * Compute derivative of this function
     * @param fx is output of this function
     * @return differentiation value at f(x)=fx, which is Square, diagonal matrix
     */
    override def derivative(fx: ScalarMatrix): ScalarMatrix = {
      // Because fx is n by 1 matrix, generate n by n matrix
      val res = ScalarMatrix $0(fx.rows, fx.rows)
      // Output is diagonal matrix, with dfi(xi)/dxi.
      (0 until fx.rows) foreach { r ⇒ {
        val expx = exp(fx(r, 0))
        res.update((r, r), (expx - 1.0) / expx)
      }
      }
      res
    }

    /**
     * Compute mapping at x
     * @param x is input scalar.
     * @return f(x)
     */
    override def apply(x: ScalarMatrix): ScalarMatrix = {
      val expx: ScalarMatrix = exp(x)
      val plus1: ScalarMatrix = expx :+ 1.0
      log(plus1)
    }
  }

  /**
   * Objective Function: Sum of Square Error
   */
  object SquaredErr extends Objective {
    /**
     * Compute derivative of this objective function
     * @param real is expected real output
     * @param output is computational output of the network
     * @return is differentiation(Gradient) vector at f(X)=output, i.e. error of each output neuron.
     */
    override def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix = output - real

    /**
     * Compute error
     * @param real is expected real output
     * @param output is computational output of the network
     * @return is the error
     */
    override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar = {
      val diff: ScalarMatrix = real - output
      val sqdiff: ScalarMatrix = diff :^ 2.0
      0.5 * sum(sqdiff)
    }
  }

  /**
   * Objective Function: Sum of Cross-Entropy (Logistic)
   */
  object CrossEntropyErr extends Objective {
    /**
     * Entropy function
     */
    val entropy = (r: Scalar, o: Scalar) ⇒ (if (r != 0.0) -r * Math.log(o) else 0.0) + (if (r != 1.0) -(1.0 - r) * Math.log(1.0 - o) else 0.0)

    /**
     * Derivative of Entropy function
     */
    val entropyDiff = (r: Scalar, o: Scalar) ⇒ (r - o) / (o * (o - 1.0))

    /**
     * Compute derivative of this objective function
     * @param real is expected real output
     * @param output is computational output of the network
     * @return is differentiation(Gradient) vector at f(X)=output, i.e. error of each output neuron.
     */
    override def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix =
      DenseMatrix.tabulate(real.rows, real.cols)((r, c) ⇒ entropyDiff(real(r, c), output(r, c)))

    /**
     * Compute error
     * @param real is expected real output
     * @param output is computational output of the network
     * @return is the error
     */
    override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar =
      sum(DenseMatrix.tabulate(real.rows, real.cols)((r, c) ⇒ entropy(real(r, c), output(r, c))))
  }

}
