package kr.ac.kaist.ir.deep.layer

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.fn.act.Activation
import play.api.libs.json.{JsObject, Json}

/**
 * Layer: Basic, Fully-connected Layer
 * @param IO is a pair of input & output, such as 2 -> 3
 * @param act is an activation function to be applied
 * @param w is initial weight matrix for the case that it is restored from JSON (default: null)
 * @param b is inital bias matrix for the case that it is restored from JSON (default: null)
 */
class BasicLayer(IO: (Int, Int),
                 protected override val act: Activation,
                 w: ScalarMatrix = null,
                 b: ScalarMatrix = null)
  extends Layer {
  /** Number of Fan-ins */
  protected val fanIn = IO._1
  /** Number of Fan-outs */
  protected val fanOut = IO._2
  /** Initialize weight */
  protected val weight = if (w != null) w else act.initialize(fanIn, fanOut)
  protected val bias = if (b != null) b else act.initialize(fanIn, fanOut, fanOut, 1)
  /** Weight-Update Accumulator */
  protected val delta = ScalarMatrix $0(fanOut, fanIn)
  protected val dbias = ScalarMatrix $0(fanOut, 1)

  /**
   * Forward computation
   * @param x of input matrix
   * @return output matrix
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val wx: ScalarMatrix = weight * x
    val wxb: ScalarMatrix = wx + bias
    act(wxb)
  }

  /**
   * Translate this layer into JSON object (in Play! framework)
   * @return JSON object describes this layer
   */
  override def toJSON: JsObject = Json.obj(
    "type" → "BasicLayer",
    "in" → fanIn,
    "out" → fanOut,
    "act" → act.getClass.getSimpleName,
    "weight" → weight.to2DSeq,
    "bias" → bias.to2DSeq
  )

  /**
   * weights for update
   * @return weights
   */
  override def W: Seq[ScalarMatrix] = Seq(weight, bias)

  /**
   * accumulated delta values
   * @return delta-weight
   */
  override def dW: Seq[ScalarMatrix] = Seq(delta, dbias)

  /**
   * <p>Backward computation.</p>
   *
   * <p>
   * Let this layer have function F composed with function <code> X(x) = W.x + b </code>
   * and higher layer have function G.
   * </p>
   *
   * <p>
   * Weight is updated with: <code>dG/dW</code>
   * and propagate <code>dG/dx</code>
   * </p>
   *
   * <p>
   * For the computation, we only used denominator layout. (cf. Wikipedia Page of Matrix Computation)
   * For the computation rules, see "Matrix Cookbook" from MIT.
   * </p>
   *
   * @param error to be propagated ( <code>dG / dF</code> is propagated from higher layer )
   * @param input of this layer (in this case, <code>x = entry of dX / dw</code>)
   * @param output of this layer (in this case, <code>y</code>)
   * @return propagated error (in this case, <code>dG/dx</code> )
   */
  protected[deep] override def !(error: ScalarMatrix, input: ScalarMatrix, output: ScalarMatrix): ScalarMatrix = {
    // fanOut × fanOut matrix (Numerator/Denominator Layout)
    val dFdX = act.derivative(output)

    /*
     * Chain Rule : dG/dX_ij = tr[ ( dG/dF ).t * dF/dX_ij ].
     *
     * Note 1. X, dG/dF, dF/dX_ij are row vectors. Therefore tr(.) can be omitted.
     * Note 2. dF/dX_ij is a column vector with all zero but (i, 1) = (i, i)-entry of dFdX.
     *
     * Thus, dG/dX = [ (dG/dF).t * dF/dX ].t, because [...] is 1 × fanOut matrix.
     * Therefore dG/dX = dF/dX * dG/dF, because dF/dX is symmetric in our case.
     */
    val dGdX: ScalarMatrix = dFdX * error

    /*
     * Chain Rule : dG/dW_ij = tr[ ( dG/dX ).t * dX/dW_ij ].
     *
     * dX/dW_ij is a fan-Out dimension column vector with all zero but (i, 1) = X_j.
     * Thus, tr(.) can be omitted, and dG/dW_ij = (dX/dW_ij).t * dG/dX
     * Then {j-th column of dG/dW} = X_j * dG/dX = dG/dX * X_j.
     *
     * Therefore dG/dW = dG/dX * X.t
     */
    val dGdW: ScalarMatrix = dGdX * input.t
    delta += dGdW

    // For bias, input is always 1. We only need dG/dX
    dbias += dGdX

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