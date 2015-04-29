package kr.ac.kaist.ir.deep.fn

import breeze.linalg.{DenseMatrix, diag}
import breeze.numerics._

/**
 * __Trait__ that describes an activation function for '''each layer'''
 *
 * Because these activation functions can be shared, we recommend to make inherited one as an object.
 */
trait Activation extends (ScalarMatrix ⇒ ScalarMatrix) with Serializable {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, diagonal matrix__
   */
  def derivative(fx: ScalarMatrix): ScalarMatrix

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  def apply(x: ScalarMatrix): ScalarMatrix

  /**
   * Initialize the weight matrix
   *
   * @param fanIn the number of __fan-in__ ''i.e. the number of neurons in previous layer''
   * @param fanOut the number of __fan-out__ ''i.e. the number of neurons in next layer''
   * @param rows the number of __rows of resulting matrix__ `(Default 0)`
   * @param cols the number of __cols of resulting matrix__ `(Default 0)`
   * @return the initialized weight matrix
   */
  def initialize(fanIn: Int, fanOut: Int, rows: Int = 0, cols: Int = 0): ScalarMatrix = {
    val pmMatx: ScalarMatrix = ScalarMatrix.of(if (rows > 0) rows else fanOut, if (cols > 0) cols else fanIn) :* 1e-2f
    pmMatx :+ 1e-2f
  }
}

/**
 * __Activation Function__: Hard version of Sigmoid
 *
 * @note `sigmoid(x) = 1 / [exp(-x) + 1]`, hard version approximates tanh as piecewise linear function
 *       (derived from relationship between tanh & sigmoid, and tanh & hard tanh.)
 * @example
 * {{{val fx = HardSigmoid(0.0)
 *          val diff = HardSigmoid.derivative(fx) }}}
 */
object HardSigmoid extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, diagonal matrix__
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = ScalarMatrix $0(fx.rows, fx.rows)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    var r = 0
    while (r < fx.rows) {
      val x = fx(r, 0)
      if (x > 0.0f && x < 1.0f)
        res.update(r, r, 0.25f)
      // else res.update((r, r), 0.0f) [Already initialized as zero]
      r += 1
    }
    res
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val res = ScalarMatrix $0(x.rows, x.cols)

    val rows = x.rows
    val cols = x.cols
    var r, c = 0

    while (r < rows) {
      c = 0
      while (c < cols) {
        val v = x(r, c)
        // if (v < -2) res.update(r, c, 0.0f) [Already initailized as zero]
        if (v > 2) res.update(r, c, 1.0f)
        else res.update(r, c, 0.25f * v + 0.5f)
        c += 1
      }
      r += 1
    }
    res
  }
}

/**
 * __Activation Function__: Hard version of Tanh (Hyperbolic Tangent)
 *
 * @note `tanh(x) = sinh(x) / cosh(x)`, hard version approximates tanh as piecewise linear function.
 * @example
 * {{{val fx = HardTanh(0.0)
 *          val diff = HardTanh.derivative(fx) }}}
 */
object HardTanh extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, diagonal matrix__
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = ScalarMatrix $0(fx.rows, fx.rows)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    var r = 0
    while (r < fx.rows) {
      val x = fx(r, 0)
      if (x < 1.0f && x > -1.0f)
        res.update(r, r, 1.0f)
      // else res.update(r, r, 0.0f) [Already initalized as zero]
      r += 1
    }
    res
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val res = x.copy
    val rows = x.rows
    val cols = x.cols
    var r, c = 0

    while (r < rows) {
      c = 0
      while (c < cols) {
        val v = x(r, c)
        if (v < -1) res.update(r, c, -1.0f)
        else if (v > 1) res.update(r, c, 1.0f)
        c += 1
      }
      r += 1
    }
    res
  }
}

/**
 * __Activation Function__: Tanh (Hyperbolic Tangent)
 *
 * @note `tanh(x) = sinh(x) / cosh(x)`
 * @example
 * {{{val fx = HyperbolicTangent(0.0)
 *         val diff = HyperbolicTangent.derivative(fx) }}}
 */
object HyperbolicTangent extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, diagonal matrix__
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = {
    // Output is diagonal matrix, with dfi(xi)/dxi.
    val dVec: ScalarMatrix = 1.0f - (fx :* fx)
    diag(dVec.toDenseVector)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = tanh(x)

  /**
   * Initialize the weight matrix
   *
   * @param fanIn the number of __fan-in__ ''i.e. the number of neurons in previous layer''
   * @param fanOut the number of __fan-out__ ''i.e. the number of neurons in next layer''
   * @param rows the number of __rows of resulting matrix__ `(Default 0)`
   * @param cols the number of __cols of resulting matrix__ `(Default 0)`
   * @return the initialized weight matrix
   */
  override def initialize(fanIn: Int, fanOut: Int, rows: Int = 0, cols: Int = 0): ScalarMatrix = {
    val range = Math.sqrt(6.0 / (fanIn + fanOut)).toFloat
    val pmMatx: ScalarMatrix = ScalarMatrix.of(if (rows > 0) rows else fanOut, if (cols > 0) cols else fanIn) :- 0.5f
    pmMatx :* (2.0f * range)
  }
}

/**
 * __Activation Function__: Linear
 *
 * @note `linear(x) = x`
 * @example
  * {{{val fx = Linear(0.0)
 *                   val diff = Linear.derivative(fx)}}}
 */
object Linear extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, diagonal matrix__
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = DenseMatrix.eye[Scalar](fx.rows)

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = x.copy
}

/**
 * __Activation Function__: Rectifier
 *
 * @note `rectifier(x) = x if x > 0, otherwise 0`
 * @example
 * {{{val fx = Rectifier(0.0)
 *         val diff = Rectifier.derivative(fx)}}}
 */
object Rectifier extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, diagonal matrix__
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = ScalarMatrix $0(fx.rows, fx.rows)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    var r = 0
    while (r < fx.rows) {
      val x = fx(r, 0)
      if (x > 0)
        res.update(r, r, 1.0f)
      //else res.update(r, r, 0.0f) [Already Initialized as zero]
      r += 1
    }
    res
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val res = x.copy
    val rows = x.rows
    val cols = x.cols
    var r, c = 0

    while (r < rows) {
      c = 0
      while (c < cols) {
        if (x(r, c) < 0) res.update(r, c, 0.0f)
        c += 1
      }
      r += 1
    }
    res
  }
}

/**
 * __Activation Function__: Sigmoid function
 *
 * @note {{{sigmoid(x) = 1 / [exp(-x) + 1]}}}
 * @example
 * {{{val fx = Sigmoid(0.0)
 *         val diff = Sigmoid.derivative(fx)}}}
 */
object Sigmoid extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, diagonal matrix__
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = {
    // Output is diagonal matrix, with dfi(xi)/dxi.
    val dVec: ScalarMatrix = (1.0f - fx) :* fx
    diag(dVec.toDenseVector)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val expv: ScalarMatrix = exp(x)
    val exp1: ScalarMatrix = expv :+ 1.0f
    1.0f / exp1
  }

  /**
   * Initialize the weight matrix
   *
   * @param fanIn the number of __fan-in__ ''i.e. the number of neurons in previous layer''
   * @param fanOut the number of __fan-out__ ''i.e. the number of neurons in next layer''
   * @param rows the number of __rows of resulting matrix__ `(Default 0)`
   * @param cols the number of __cols of resulting matrix__ `(Default 0)`
   * @return the initialized weight matrix
   */
  override def initialize(fanIn: Int, fanOut: Int, rows: Int = 0, cols: Int = 0): ScalarMatrix = {
    val range = (Math.sqrt(6.0 / (fanIn + fanOut)) * 4.0).toFloat
    val pmMatx: ScalarMatrix = ScalarMatrix.of(if (rows > 0) rows else fanOut, if (cols > 0) cols else fanIn) :- 0.5f
    pmMatx :* (2.0f * range)
  }
}

/**
 * __Activation Function__: Softplus
 *
 * @note `softplus(x) = log[1 + exp(x)]`
 * @example
 * {{{val fx = Softplus(0.0)
 *         val diff = Softplus.derivative(fx)}}}
 */
object Softplus extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, diagonal matrix__
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = {
    // Output is diagonal matrix, with dfi(xi)/dxi.
    val expv: ScalarMatrix = exp(fx)
    val exp1: ScalarMatrix = expv - 1.0f
    val dVec: ScalarMatrix = exp1 / expv
    diag(dVec.toDenseVector)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val expx = exp(x)
    val plus1 = expx :+ 1.0f
    log(plus1)
  }
}