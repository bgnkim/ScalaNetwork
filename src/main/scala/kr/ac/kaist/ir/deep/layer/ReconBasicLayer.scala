package kr.ac.kaist.ir.deep.layer

import kr.ac.kaist.ir.deep.fn._
import play.api.libs.json.JsObject

/**
 * __Layer__ : Reconstructable Basic Layer
 *
 * @param IO is a pair of input & output, such as 2 -> 3
 * @param act is an activation function to be applied
 * @param w is initial weight matrix for the case that it is restored from JSON (default: null)
 * @param b is inital bias matrix for the case that it is restored from JSON (default: null)
 * @param rb is initial reconstruct bias matrix for the case that it is restored from JSON (default: null)
 */
class ReconBasicLayer(IO: (Int, Int),
                      act: Activation,
                      w: ScalarMatrix = null,
                      b: ScalarMatrix = null,
                      rb: ScalarMatrix = null)
  extends BasicLayer(IO, act, w, b) with Reconstructable {
  protected lazy val reBias = if (rb != null) rb else act initialize(fanIn, fanOut, fanIn, 1)
  protected lazy val drBias = ScalarMatrix $0(fanIn, 1)
  /**
   * weights for update
   *
   * @return weights
   */
  override val W: IndexedSeq[ScalarMatrix] = IndexedSeq(reBias, weight, bias)
  /**
   * accumulated delta values
   *
   * @return delta-weight
   */
  override val dW: IndexedSeq[ScalarMatrix] = IndexedSeq(drBias, delta, dbias)

  /**
   * Sugar: reconstruction
   *
   * @param x hidden layer output matrix
   * @return tuple of reconstruction output
   */
  override def decodeBy_:(x: ScalarMatrix): ScalarMatrix = {
    val wx: ScalarMatrix = weight.t[ScalarMatrix, ScalarMatrix] * x
    val wxb: ScalarMatrix = wx + reBias
    act(wxb)
  }

  /**
   * Translate this layer into JSON object (in Play! framework)
   *
   * @return JSON object describes this layer
   */
  override def toJSON: JsObject = super.toJSON + ("reconst_bias" → reBias.to2DSeq)

  /**
   * Backpropagation of reconstruction. For the information about backpropagation calculation, see [[kr.ac.kaist.ir.deep.layer.Layer]]
   *
   * @param error error matrix to be propagated
   * @param input input of this layer
   * @param output final reconstruction output of this layer
   * @return propagated error
   */
  protected[deep] override def decodeUpdateBy(error: ScalarMatrix, input: ScalarMatrix, output: ScalarMatrix): ScalarMatrix = {
    // Recon × Recon matrix
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
    delta += dGdW.t // Because we used transposed weight for reconstruction, we need to transpose it.

    // For bias, input is always 1. We only need dG/dX
    drBias += dGdX

    /*
     * Chain Rule : dG/dx_ij = tr[ ( dG/dX ).t * dX/dx_ij ].
     *
     * X is column vector. Thus j is always 1, so dX/dx_i is a W_?i.
     * Hence dG/dx_i = tr[ (dG/dX).t * dX/dx_ij ] = (W_?i).t * dG/dX.
     *
     * Thus dG/dx = W.t * dG/dX
     */
    val dGdx: ScalarMatrix = weight * dGdX // Because we used transposed weight for reconstruction.
    dGdx
  }
}