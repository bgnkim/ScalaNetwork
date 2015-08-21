package kr.ac.kaist.ir.deep.layer

import kr.ac.kaist.ir.deep.fn._
import play.api.libs.json.{JsObject, Json}

/**
 * __Layer__: Basic, Fully-connected Layer
 *
 * @param IO a pair of __input & output__, such as 2 -> 3
 * @param act an __activation function__ to be applied
 * @param w initial weight matrix for the case that it is restored from JSON `(default: null)`
 * @param b inital bias matrix for the case that it is restored from JSON `(default: null)`
 */
class BasicLayer(IO: (Int, Int),
                 protected override val act: Activation,
                 w: ScalarMatrix = null,
                 b: ScalarMatrix = null)
  extends Layer {
  require(act != null, "Activation function must not be null.")
  require(IO._1 > 0, "Input dimension must be greater than 0")
  require(IO._2 > 0, "Output dimension must be greater than 0")

  /** Number of Fan-ins */
  protected final val fanIn = IO._1
  /** Number of output */
  protected final val fanOut = IO._2
  /* Initialize weight */
  protected final val weight = if (w != null) w else act.initialize(fanIn, fanOut)
  protected final val bias = if (b != null) b else act.initialize(fanIn, fanOut, fanOut, 1)
  /** weights for update */
  override val W: IndexedSeq[ScalarMatrix] = IndexedSeq(weight, bias)

  /**
   * Forward computation
   *
   * @param x input matrix
   * @return output matrix
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val wx: ScalarMatrix = weight * x
    val wxb: ScalarMatrix = wx + bias
    act(wxb)
  }

  /**
   * Translate this layer into JSON object (in Play! framework)
   *
   * @return JSON object describes this layer
   */
  override def toJSON: JsObject = Json.obj(
    "type" → "BasicLayer",
    "in" → fanIn,
    "out" → fanOut,
    "act" → act.toJSON,
    "weight" → weight.to2DSeq,
    "bias" → bias.to2DSeq
  )

  /**
   * <p>Backward computation.</p>
   *
   * @note <p>
   *       Let this layer have function F composed with function <code> X(x) = W.x + b </code>
   *       and higher layer have function G.
   *       </p>
   *
   *       <p>
   *       Weight is updated with: <code>dG/dW</code>
   *       and propagate <code>dG/dx</code>
   *       </p>
   *
   *       <p>
   *       For the computation, we only used denominator layout. (cf. Wikipedia Page of Matrix Computation)
   *       For the computation rules, see "Matrix Cookbook" from MIT.
   *       </p>
   *
   * @param delta Sequence of delta amount of weight. The order must be the reverse of [[W]]
   *              In this function, (bias :: weight) ::: lowerStack
   *              Thus dWeight is app
   * @param error to be propagated ( <code>dG / dF</code> is propagated from higher layer )
   * @return propagated error (in this case, <code>dG/dx</code> )
   */
  def updateBy(delta: Iterator[ScalarMatrix], error: ScalarMatrix): ScalarMatrix = {
    /*
     * Chain Rule : dG/dX_ij = tr[ ( dG/dF ).t * dF/dX_ij ].
     *
     * Note 1. X, dG/dF, dF/dX_ij are row vectors. Therefore tr(.) can be omitted.
     *
     * Thus, dG/dX = [ (dG/dF).t * dF/dX ].t, because [...] is 1 × fanOut matrix.
     * Therefore dG/dX = dF/dX * dG/dF, because dF/dX is symmetric in our case.
     */
    val dGdX: ScalarMatrix = dFdX * error

    // For bias, input is always 1. We only need dG/dX
    delta.next += dGdX

    /*
     * Chain Rule : dG/dW_ij = tr[ ( dG/dX ).t * dX/dW_ij ].
     *
     * dX/dW_ij is a fan-Out dimension column vector with all zero but (i, 1) = X_j.
     * Thus, tr(.) can be omitted, and dG/dW_ij = (dX/dW_ij).t * dG/dX
     * Then {j-th column of dG/dW} = X_j * dG/dX = dG/dX * X_j.
     *
     * Therefore dG/dW = dG/dX * X.t
     */
    val dGdW: ScalarMatrix = dGdX * X.t
    delta.next += dGdW

    /*
     * Chain Rule : dG/dx_ij = tr[ ( dG/dX ).t * dX/dx_ij ].
     *
     * X is column vector. Thus j is always 1, so dX/dx_i is a W_?i.
     * Hence dG/dx_i = tr[ (dG/dX).t * dX/dx_ij ] = (W_?i).t * dG/dX.
     *
     * Thus dG/dx = W.t * dG/dX
     */
    val dGdx: ScalarMatrix = weight.t * dGdX
    dGdx
  }
}