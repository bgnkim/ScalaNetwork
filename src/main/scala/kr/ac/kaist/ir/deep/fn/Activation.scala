package kr.ac.kaist.ir.deep.fn

import breeze.linalg.{DenseMatrix, diag, sum}
import breeze.numerics._
import play.api.libs.json.{JsObject, JsValue, Json}

import scala.annotation.tailrec
import scala.reflect.runtime._

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
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
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
   * Serialize Activation function into String.
   * @note If this is an "object", do not modify this function.
   *       This does not supports Activation Operations defined outside of this package.
   * @return JSON object states this function
   */
  def toJSON: JsObject = Json.obj("class" → this.getClass.getCanonicalName)

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
 * Companion Object of Activation.
 */
object Activation {
  @transient val runtimeMirror = universe.synchronized(universe.runtimeMirror(getClass.getClassLoader))

  /**
   * Reconstruct Activation function from given JSON value.
   * @param obj JSON value to be reconstructed
   * @return Activation reconstructed from JSON
   */
  def apply(obj: JsValue): Activation = {
    (obj \ "function").asOpt[String] match {
      case Some("scale") ⇒
        val base = apply(obj \ "base")
        val x = (obj \ "X").as[Float]
        val y = (obj \ "Y").as[Float]
        base *(x, y)
      case Some("translation") ⇒
        val base = apply(obj \ "base")
        val x = (obj \ "X").as[Float]
        val y = (obj \ "Y").as[Float]
        base +(x, y)
      case Some("add") ⇒
        val base = apply(obj \ "base")
        val args = (obj \ "args").as[Seq[JsValue]].map(apply)
        base.+(args: _*)
      case _ ⇒
        val str = (obj \ "class").asOpt[String] match {
          case Some(x) ⇒ x
          case _ ⇒ "kr.ac.kaist.ir.deep.fn." + obj.as[String]
        }
        universe.synchronized {
          val module = runtimeMirror.staticModule(str)
          runtimeMirror.reflectModule(module).instance.asInstanceOf[Activation]
        }
    }
  }
}

/**
 * __Activation Function__: Hard version of Sigmoid
 *
 * @note `sigmoid(x) = 1 / [exp(-x) + 1]`, hard version approximates tanh as piecewise linear function
 *       (derived from relationship between tanh & sigmoid, and tanh & hard tanh.)
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = HardSigmoid(0.0)
 *          val diff = HardSigmoid.derivative(fx) }}}
 */
object HardSigmoid extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = ScalarMatrix $0(fx.rows, fx.rows)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    derivCoord(fx, res, fx.rows - 1)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val res = ScalarMatrix $0(x.rows, x.cols)
    applyCoord(x, res, x.rows - 1, x.cols - 1)
  }

  @tailrec
  private def derivCoord(fx: ScalarMatrix, res: ScalarMatrix, r: Int): ScalarMatrix =
    if (r >= 0) {
      val x = fx(r, 0)
      if (x > 0.0f && x < 1.0f)
        res.update(r, r, 0.25f)
      // else res.update((r, r), 0.0f) [Already initialized as zero]
      derivCoord(fx, res, r - 1)
    } else
      res

  @tailrec
  private def applyCoord(x: ScalarMatrix, res: ScalarMatrix, r: Int, c: Int): ScalarMatrix =
    if (r >= 0) {
      val v = x(r, c)
      // if (v < -2) res.update(r, c, 0.0f) [Already initailized as zero]
      if (v > 2) res.update(r, c, 1.0f)
      else res.update(r, c, 0.25f * v + 0.5f)

      if (c > 0)
        applyCoord(x, res, r, c - 1)
      else
        applyCoord(x, res, r - 1, x.cols - 1)
    } else
      res
}

/**
 * __Activation Function__: Hard version of Tanh (Hyperbolic Tangent)
 *
 * @note `tanh(x) = sinh(x) / cosh(x)`, hard version approximates tanh as piecewise linear function.
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = HardTanh(0.0)
 *          val diff = HardTanh.derivative(fx) }}}
 */
object HardTanh extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = ScalarMatrix $0(fx.rows, fx.rows)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    derivCoord(fx, res, fx.rows - 1)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val res = x.copy
    applyCoord(x, res, x.rows - 1, x.cols - 1)
  }

  @tailrec
  private def derivCoord(fx: ScalarMatrix, res: ScalarMatrix, r: Int): ScalarMatrix =
    if (r >= 0) {
      val x = fx(r, 0)
      if (x < 1.0f && x > -1.0f)
        res.update(r, r, 1.0f)
      // else res.update(r, r, 0.0f) [Already initalized as zero]
      derivCoord(fx, res, r - 1)
    } else
      res

  @tailrec
  private def applyCoord(x: ScalarMatrix, res: ScalarMatrix, r: Int, c: Int): ScalarMatrix =
    if (r >= 0) {
      val v = x(r, c)
      if (v < -1) res.update(r, c, -1.0f)
      else if (v > 1) res.update(r, c, 1.0f)

      if (c > 0)
        applyCoord(x, res, r, c - 1)
      else
        applyCoord(x, res, r - 1, x.cols - 1)
    } else
      res
}

/**
 * __Activation Function__: Tanh (Hyperbolic Tangent)
 *
 * @note `tanh(x) = sinh(x) / cosh(x)`
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = HyperbolicTangent(0.0)
 *         val diff = HyperbolicTangent.derivative(fx) }}}
 */
object HyperbolicTangent extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
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
 *       We assumed the input of activation is a row vector.
 * @example
  * {{{val fx = Linear(0.0)
 *                   val diff = Linear.derivative(fx)}}}
 */
object Linear extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
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
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = Rectifier(0.0)
 *         val diff = Rectifier.derivative(fx)}}}
 */
object Rectifier extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = {
    // Because fx is n by 1 matrix, generate n by n matrix
    val res = ScalarMatrix $0(fx.rows, fx.rows)
    // Output is diagonal matrix, with dfi(xi)/dxi.
    derivCoord(fx, res, fx.rows - 1)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val res = x.copy
    applyCoord(x, res, x.rows - 1, x.cols - 1)
  }

  @tailrec
  private def derivCoord(fx: ScalarMatrix, res: ScalarMatrix, r: Int): ScalarMatrix =
    if (r >= 0) {
      val x = fx(r, 0)
      if (x > 0)
        res.update(r, r, 1.0f)
      //else res.update(r, r, 0.0f) [Already Initialized as zero]
      derivCoord(fx, res, r - 1)
    } else
      res

  @tailrec
  private def applyCoord(x: ScalarMatrix, res: DenseMatrix[Scalar], r: Int, c: Int): ScalarMatrix =
    if (r >= 0) {
      if (x(r, c) < 0) res.update(r, c, 0.0f)

      if (c > 0) applyCoord(x, res, r, c - 1)
      else applyCoord(x, res, r - 1, x.cols - 1)
    } else
      res
}

/**
 * __Activation Function__: Sigmoid function
 *
 * @note {{{sigmoid(x) = 1 / [exp(-x) + 1]}}}
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = Sigmoid(0.0)
 *         val diff = Sigmoid.derivative(fx)}}}
 */
object Sigmoid extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
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
 * __Activation Function__: Softmax function
 *
 * @note {{{softmax(x)_i = exp(x_i) / sum(exp(x_i))}}}
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = Softmax(0.0)
 *          val diff = Softmax.derivative(fx)}}}
 */
object Softmax extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
   */
  override def derivative(fx: ScalarMatrix): ScalarMatrix = {
    val res: ScalarMatrix = ScalarMatrix $0(fx.rows, fx.rows)

    // Note that (i, j)-entry of deriviative is dF_i / dX_j
    // and dF_i / dX_j = F(i) * (Delta_ij - F(j)).
    initDeriv(fx, res, fx.rows - 1)
    derivCoord(fx, res, res.rows - 1, res.cols - 1)
  }

  /**
   * Compute mapping for `x`
   *
   * @param x the __input__ matrix. ''Before application, input should be summed already.''
   * @return value of `f(x)`
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val expv: ScalarMatrix = exp(x)
    val normalize: Scalar = sum(expv)
    expv :/= normalize
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

  @tailrec
  private def initDeriv(fx: ScalarMatrix, res: ScalarMatrix, r: Int): Unit =
    if (r >= 0) {
      res(r, ::) := fx(r, 0)
      initDeriv(fx, res, r - 1)
    }

  @tailrec
  private def derivCoord(fx: ScalarMatrix, res: ScalarMatrix, r: Int, c: Int): ScalarMatrix =
    if (r >= 0) {
      val dfdx = (if (r == c) 1 else 0) - fx(c, 0)
      res.update(r, c, res(r, c) * dfdx)

      if (c > 0) derivCoord(fx, res, r, c - 1)
      else derivCoord(fx, res, r - 1, fx.rows - 1)
    } else
      res
}

/**
 * __Activation Function__: Softplus
 *
 * @note `softplus(x) = log[1 + exp(x)]`
 *       We assumed the input of activation is a row vector.
 * @example
 * {{{val fx = Softplus(0.0)
 *         val diff = Softplus.derivative(fx)}}}
 */
object Softplus extends Activation {
  /**
   * Compute differentiation value of this function at `f(x) = fx`
   *
   * @param fx the __output__ of this function
   * @return differentiation value at `f(x) = fx`, which should be an __square, symmetric matrix__
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