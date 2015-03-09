package kr.ac.kaist.ir.deep.fn

import breeze.linalg.{DenseMatrix, sum}
import breeze.numerics._

/**
 * __Trait__ that describes an objective function for '''entire network'''
 *
 * Because these objective functions can be shared, we recommend to make inherited one as an object. 
 */
trait Objective extends ((ScalarMatrix, ScalarMatrix) ⇒ Scalar) with Serializable {
  /**
   * Compute differentiation value of this objective function at `x = r - o`
   *
   * @param real the expected __real output__, `r`
   * @param output the computed __output of the network__, `o`
   * @return differentiation value at `f(x)=fx`, which is __a column vector__
   */
  def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix

  /**
   * Compute error (loss)
   *
   * @param real the expected __real output__
   * @param output the computed __output of the network__
   * @return the error
   */
  override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar
}


/**
 * __Objective Function__: Cosine Similarity Error
 *
 * This has a heavy computation. If you want to use lighter one, use [[DotProductErr]]
 *
 * @note This function returns 1 - cosine similarity, i.e. cosine dissimiarlity.
 *
 * @example
 * {{{val output = net(input)
 *         val err = CosineErr(real, output)
 *         val diff = CosineErr.derivative(real, output)
 * }}}
 */
object CosineErr extends Objective {
  /**
   * Compute differentiation value of this objective function w.r.t output o
   *
   * @param real the expected __real output__, `r`
   * @param output the computed __output of the network__, `o`
   * @return differentiation value at `f(x)=fx`, which is __a column vector__
   */
  override def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix = {
    val dotValue: Scalar = sum(real :* output)

    val lenReal = len(real)
    val lenOut = len(output)
    val lenOutSq = lenOut * lenOut
    // Denominator of derivative is len(real) * len(output)^3
    val denominator = lenReal * lenOut * lenOutSq

    DenseMatrix.tabulate(output.rows, output.cols) {
      (r, c) ⇒
        val x = output(r, c)
        val a = real(r, c)
        // The nominator of derivative of cosine similarity is,
        // a(lenOut^2 - x^2) - x(dot - a*x)
        // = a*lenOut^2 - x*dot
        val nominator = a * lenOutSq - x * dotValue

        // We need derivative of 1 - cosine.
        -(nominator / denominator)
    }
  }

  /**
   * Compute error (loss)
   *
   * @param real the expected __real output__
   * @param output the computed __output of the network__
   * @return the error
   */
  override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar = {
    val norm = len(real) * len(output)
    val dotValue: Scalar = sum(real :* output)
    1.0f - (dotValue / norm)
  }

  /**
   * length of given matrix
   * @param matrix matrix
   * @return length = sqrt(sum(pow(:, 2)))
   */
  private def len(matrix: ScalarMatrix): Scalar = {
    Math.sqrt(sum(pow(matrix, 2.0f))).toFloat
  }
}

/**
 * __Objective Function__: Sum of Cross-Entropy (Logistic)
 *
 * @note This objective function prefer 0/1 output
 * @example
 * {{{val output = net(input)
 *         val err = CrossEntropyErr(real, output)
 *         val diff = CrossEntropyErr.derivative(real, output)
 * }}}
 */
object CrossEntropyErr extends Objective {
  /**
   * Entropy function
   */
  val entropy = (r: Scalar, o: Scalar) ⇒
    (if (r != 0.0f) -r * Math.log(o).toFloat else 0.0f) + (if (r != 1.0f) -(1.0f - r) * Math.log(1.0f - o).toFloat else 0.0f)

  /**
   * Derivative of Entropy function
   */
  val entropyDiff = (r: Scalar, o: Scalar) ⇒ (r - o) / (o * (o - 1.0f))

  /**
   * Compute differentiation value of this objective function w.r.t output o
   *
   * @param real the expected __real output__, `r`
   * @param output the computed __output of the network__, `o`
   * @return differentiation value at `f(x)=fx`, which is __a column vector__
   */
  override def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix =
    DenseMatrix.tabulate(real.rows, real.cols)((r, c) ⇒ entropyDiff(real(r, c), output(r, c)))

  /**
   * Compute error (loss)
   *
   * @param real the expected __real output__
   * @param output the computed __output of the network__
   * @return the error
   */
  override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar =
    sum(DenseMatrix.tabulate(real.rows, real.cols)((r, c) ⇒ entropy(real(r, c), output(r, c))))
}

/**
 * __Objective Function__: Dot-product Error
 *
 * @note This function computes additive inverse of dot product, i.e. dot-product dissimiarity.
 *
 * @example
 * {{{val output = net(input)
 *         val err = DotProductErr(real, output)
 *         val diff = DotProductErr.derivative(real, output)
 * }}}
 */
object DotProductErr extends Objective {
  /**
   * Compute differentiation value of this objective function w.r.t output o
   *
   * @param real the expected __real output__, `r`
   * @param output the computed __output of the network__, `o`
   * @return differentiation value at `f(x)=fx`, which is __a column vector__
   */
  override def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix = -real

  /**
   * Compute error (loss)
   *
   * @param real the expected __real output__
   * @param output the computed __output of the network__
   * @return the error
   */
  override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar = - sum(real :* output)
}

/**
 * __Objective Function__: Sum of Squared Error
 *
 * @example
 * {{{val output = net(input)
 *         val err = SquaredErr(real, output)
 *         val diff = SquaredErr.derivative(real, output)
 * }}}
 */
object SquaredErr extends Objective {
  /**
   * Compute differentiation value of this objective function w.r.t output o
   *
   * @param real the expected __real output__, `r`
   * @param output the computed __output of the network__, `o`
   * @return differentiation value at `f(x)=fx`, which is __a column vector__
   */
  override def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix = output - real

  /**
   * Compute error (loss)
   *
   * @param real the expected __real output__
   * @param output the computed __output of the network__
   * @return the error
   */
  override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar = {
    val diff = real - output
    sum(pow(diff, 2.0f))
  }
}

/**
 * __Objective Function__: Sum of Absolute Error
 *
 * @note In mathematics, L,,1,,-distance is called ''Manhattan distance.''
 *
 * @example
 * {{{val output = net(input)
 *          val err = ManhattanErr(real, output)
 *          val diff = ManhattanErr.derivative(real, output)
 * }}}
 */
object ManhattanErr extends Objective {
  /**
   * Compute differentiation value of this objective function w.r.t output o
   *
   * @param real the expected __real output__, `r`
   * @param output the computed __output of the network__, `o`
   * @return differentiation value at `f(x)=fx`, which is __a column vector__
   */
  override def derivative(real: ScalarMatrix, output: ScalarMatrix): ScalarMatrix =
    DenseMatrix.tabulate(real.rows, real.cols) {
      (r, c) ⇒
        val target = real(r, c)
        val x = output(r, c)
        if (target > x) 1.0f
        else if (target < x) -1.0f
        else 0.0f
    }

  /**
   * Compute error (loss)
   *
   * @param real the expected __real output__
   * @param output the computed __output of the network__
   * @return the error
   */
  override def apply(real: ScalarMatrix, output: ScalarMatrix): Scalar = {
    val diff = real - output
    sum(abs(diff))
  }
}