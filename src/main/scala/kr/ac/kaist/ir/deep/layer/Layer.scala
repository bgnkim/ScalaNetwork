package kr.ac.kaist.ir.deep.layer

import kr.ac.kaist.ir.deep.function._
import play.api.libs.json.JsObject

/**
 * Trait: Layer
 *
 * Layer is an instance of ScalaMatrix => ScalaMatrix function.
 * Therefore "layers" can be composed together.
 */
trait Layer extends (ScalarMatrix ⇒ ScalarMatrix) with Serializable {
  /** Activation Function */
  protected val act: Activation

  /**
   * Forward computation
   * @param x of input matrix
   * @return output matrix
   */
  override def apply(x: ScalarMatrix): ScalarMatrix

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
  protected[deep] def !(error: ScalarMatrix, input: ScalarMatrix, output: ScalarMatrix): ScalarMatrix

  /**
   * Sugar: Forward computation. Calls apply(x)
   *
   * @param x of input matrix
   * @return output matrix
   */
  protected[deep] def >>:(x: ScalarMatrix) = apply(x)

  /**
   * Translate this layer into JSON object (in Play! framework)
   * @return JSON object describes this layer
   */
  def toJSON: JsObject

  /**
   * weights for update
   * @return weights
   */
  def W: Seq[ScalarMatrix]

  /**
   * accumulated delta values
   * @return delta-weight
   */
  def dW: Seq[ScalarMatrix]
}
