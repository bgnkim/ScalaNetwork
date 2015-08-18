package kr.ac.kaist.ir.deep.layer

import breeze.linalg.sum
import breeze.numerics.{exp, pow}
import kr.ac.kaist.ir.deep.fn._
import play.api.libs.json.{JsObject, Json}

import scala.annotation.tailrec

/**
 * __Layer__ : An Radial Basis Function Layer, with Gaussian function as its radial basis.
 *
 * @param in Dimension of input
 * @param centers A Matrix of Centroids. Each column is a column vector for centroids.
 * @param canModifyCenter True if update center during training.
 * @param w Initial weight (default: null)
 */
class GaussianRBFLayer(val in: Int,
                       val centers: ScalarMatrix,
                       val canModifyCenter: Boolean = true,
                       w: ScalarMatrix = null) extends Layer {
  protected final val weight = if (w != null) w else ScalarMatrix of(centers.cols, 1)
  protected final val dWeight = ScalarMatrix of(centers.cols, 1)
  protected final val dCenter = ScalarMatrix of(centers.rows, centers.cols)
  protected final val sumCentroidEff = ScalarMatrix $1(centers.cols, 1)
  protected final val sumByRow = ScalarMatrix $1(1, in)
  override protected val act: Activation = null
  override val W: IndexedSeq[ScalarMatrix] = IndexedSeq(centers, weight)
  override val dW: IndexedSeq[ScalarMatrix] = IndexedSeq(dCenter, dWeight)

  /**
   * Translate this layer into JSON object (in Play! framework)
   * @note Please make an LayerReviver object if you're using custom layer.
   *       In that case, please specify LayerReviver object's full class name as "__reviver__,"
   *       and fill up LayerReviver.revive method.
   *
   * @return JSON object describes this layer
   */
  override def toJSON: JsObject = Json.obj(
    "type" → "GaussianRBF",
    "in" → in,
    "center" → centers.to2DSeq,
    "canModifyCenter" → canModifyCenter,
    "weight" → weight.to2DSeq
  )

  /**
   * Forward computation
   *
   * @param x input matrix
   * @return output matrix
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val sqWeight: ScalarMatrix = pow(weight, 2f) :* 2f
    exp(applyCoord(x, sqWeight, ScalarMatrix $0(centers.cols, 1), centers.cols - 1))
  }

  /**
   * <p>Backward computation.</p>
   *
   * @note <p>
   *       Let X ~ N(c_i, s_i) be the Gaussian distribution, and let N_i be the pdf of it.
   *       Then the output of this layer will be : <code>y_i = N_i(x) = exp(-[x-c_i]*[x-c_i]/[2*s_i*s_i])</code>.
   *       Call function on the higher layers as G.
   *       </p>
   *
   *       <p>
   *       Centers are updated with: <code>dG/dC_ij = dG/dN_i * dN_i/dc_ij.</code>
   *       Weights are updated with: <code>dG/dW_i = dG/dN_i * dN_i/dw_i.</code>
   *       and propagate <code>dG/dx_j = \sum_i dG/dN_i * dN_i/dx_ij.</code>
   *       </p>
   *
   * @param error to be propagated ( <code>dG / dN</code> is propagated from higher layer )
   * @return propagated error (in this case, <code>dG/dx</code> )
   */
  override protected[deep] def updateBy(error: ScalarMatrix) = {
    val multiplier: ScalarMatrix = (error :* dFdX) :/ pow(weight, 2f)

    val dGdC: ScalarMatrix = updateCoord(multiplier, ScalarMatrix.$0(centers.rows, centers.cols), centers.cols - 1)

    if (canModifyCenter) {
      dCenter += dGdC
    }

    -dGdC * sumCentroidEff
  }

  @tailrec
  private def applyCoord(x: ScalarMatrix, sqWeight: ScalarMatrix, out: ScalarMatrix, i: Int): ScalarMatrix =
    if (i >= 0) {
      val d: Scalar = sum(pow(x - centers(::, i to i), 2f))
      val in = -d / sqWeight(i, 0)

      out(i, 0) = in
      applyCoord(x, sqWeight, out, i - 1)
    } else
      out

  @tailrec
  private def updateCoord(multiplier: ScalarMatrix, dGdC: ScalarMatrix, i: Int): ScalarMatrix =
    if (i >= 0) {
      val d: ScalarMatrix = X - centers(::, i to i)

      val w = weight(i, 0)
      val m = multiplier(i, 0)

      /* Compute dNi/dCij.
       * Since Ni = exp(-|x-ci|^2/(2si^2)), dNi/dCij = (xj-cij)/si^2 * Ni.
       * Therefore dNi/dCi = (x-ci)/si^2 * Ni.
       * dG/dCi = dG/dNi * dNi/dCi.
       * Note that dNi/dX = -dNi/dCi, and dG/dX = - \sum (dG/dNi * dNi/dCi)
       */
      dGdC(::, i to i) := d * m

      /* Compute dG/dSi.
       * dNi/dSi = |x-ci|^2/si^3 * Ni.
       * dG/dSi = dG/dNi * dNi/dSi.
       */
      dWeight(i, 0) += sum(pow(d, 2f)) * (m / w)
      updateCoord(multiplier, dGdC, i - 1)
    } else
      dGdC

}
