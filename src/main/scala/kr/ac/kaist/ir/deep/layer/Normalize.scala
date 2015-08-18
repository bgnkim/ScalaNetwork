package kr.ac.kaist.ir.deep.layer

import breeze.linalg.sum
import breeze.numerics.pow
import kr.ac.kaist.ir.deep.fn._
import play.api.libs.json.{JsObject, Json}

/**
 * __Layer__ that normalizes its input.
 */
trait Normalize extends Layer {
  /**
   * weights for update
   *
   * @return weights
   */
  override val W: IndexedSeq[ScalarMatrix] = IndexedSeq.empty
  /**
   * accumulated delta values
   *
   * @return delta-weight
   */
  override val dW: IndexedSeq[ScalarMatrix] = IndexedSeq.empty
  /** Null activation */
  protected override val act = null

  /**
   * Forward computation
   *
   * @param x input matrix
   * @return output matrix
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val len = Math.sqrt(sum(pow(x, 2.0f))).toFloat
    x :/ len
  }

  /**
   * Translate this layer into JSON object (in Play! framework)
   *
   * @return JSON object describes this layer
   */
  override def toJSON: JsObject = super.toJSON ++ Json.obj("Normalize" → "")

  /**
   * <p>Backward computation.</p>
   *
   * @note Because this layer only mediates two layers, this layer just remove propagated error for unused elements. 
   *
   * @param error to be propagated ( <code>dG / dF</code> is propagated from higher layer )
   * @return propagated error (in this case, <code>dG/dx</code> )
   */
  protected[deep] override def updateBy(error: ScalarMatrix): ScalarMatrix = {
    val Xsq = pow(X, 2.0f)
    val lenSq = sum(Xsq)
    val len: Scalar = Math.sqrt(lenSq).toFloat

    // Note that length is the function of x_i.
    // Let z_i := x_i / len(x_i).
    // Then d z_i / d x_i = (len^2 - x_i^2) / len^3 = (1 - z_i^2) / len,
    //      d z_j / d x_i = - x_i * x_j / len^3 = - z_i * z_j / len
    val rows = dFdX.rows
    val dZdX = ScalarMatrix $0(rows, rows)
    var r = 0
    while (r < rows) {
      //dZ_r
      var c = 0
      while (c < rows) {
        if (r == c) {
          //dX_c
          dZdX.update(r, c, (1.0f - Xsq(r, 0) / lenSq) / len)
        } else {
          dZdX.update(r, c, (-X(r, 0) * X(c, 0)) / (len * lenSq))
        }
        c += 1
      }
      r += 1
    }

    // un-normalize the error
    dZdX * error
  }
}