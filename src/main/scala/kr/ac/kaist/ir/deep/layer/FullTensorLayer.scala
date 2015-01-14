package kr.ac.kaist.ir.deep.layer

import kr.ac.kaist.ir.deep.function.{Activation, ScalarMatrix}

/**
 * Layer: Basic, Fully-connected Rank 3 Tensor Layer.
 *
 * <pre>
 * v0 = a column vector
 * Q = Rank 3 Tensor with size out, in × in is its entry.
 * L = Rank 3 Tensor with size out, 1 × in is its entry.
 * b = out × 1 matrix.
 *
 * output = f( v0'.Q.v0 + L.v0 + b )
 * </pre>
 *
 * @param IO is a tuple of the number of input and output, i.e. (2 → 4)
 * @param act is an activation function to be applied
 * @param quad is initial quadratic-level weight matrix Q for the case that it is restored from JSON (default: Seq())
 * @param lin is initial linear-level weight matrix L for the case that it is restored from JSON (default: Seq())
 * @param const is initial bias weight matrix b for the case that it is restored from JSON (default: null)
 */
class FullTensorLayer(IO: (Int, Int),
                      protected override val act: Activation,
                      quad: Seq[ScalarMatrix] = Seq(),
                      lin: Seq[ScalarMatrix] = Seq(),
                      const: ScalarMatrix = null)
  extends Rank3TensorLayer(IO._2, act, quad, lin, const) {
  /** Number of Fan-ins */
  override protected val fanInA: Int = IO._1
  override protected val fanInB: Int = IO._1
  override protected val fanIn: Int = IO._1

  /**
   * Retrieve first input
   * @param x to be separated
   * @return first input
   */
  protected override def in1(x: ScalarMatrix): ScalarMatrix = x

  /**
   * Retrive second input
   * @param x to be separated
   * @return second input
   */
  protected override def in2(x: ScalarMatrix): ScalarMatrix = x
}
