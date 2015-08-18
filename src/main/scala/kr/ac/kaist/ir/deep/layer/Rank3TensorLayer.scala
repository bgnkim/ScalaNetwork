package kr.ac.kaist.ir.deep.layer

import kr.ac.kaist.ir.deep.fn._

import scala.collection.mutable.ArrayBuffer

/**
 * __Layer__: Basic, Fully-connected Rank 3 Tensor Layer.
 *
 * @note <pre>
 *       v0 = a column vector concatenate v2 after v1 (v11, v12, ... v1in1, v21, ...)
 *       Q = Rank 3 Tensor with size out, in1 × in2 is its entry.
 *       L = Rank 3 Tensor with size out, 1 × (in1 + in2) is its entry.
 *       b = out × 1 matrix.
 *
 *       output = f( v1'.Q.v2 + L.v0 + b )
 *       </pre>
 *
 * @param fanIns is the number of input. (vector1, vector2, entire).
 * @param fanOut is the number of output
 * @param act is an activation function to be applied
 * @param quad is initial quadratic-level weight matrix Q for the case that it is restored from JSON (default: Seq())
 * @param lin is initial linear-level weight matrix L for the case that it is restored from JSON (default: null)
 * @param const is initial bias weight matrix b for the case that it is restored from JSON (default: null)
 */
abstract class Rank3TensorLayer(protected val fanIns: (Int, Int, Int),
                                protected val fanOut: Int,
                                protected override val act: Activation,
                                quad: Seq[ScalarMatrix] = Seq(),
                                lin: ScalarMatrix = null,
                                const: ScalarMatrix = null)
  extends Layer {
  /* Number of Fan-ins */
  protected final val fanInA = fanIns._1
  protected final val fanInB = fanIns._2
  protected final val fanIn = fanIns._3
  /* Initialize weight */
  protected final val quadratic: ArrayBuffer[ScalarMatrix] = ArrayBuffer()
  protected final val linear: ScalarMatrix = if (lin != null) lin else act.initialize(fanIn, fanOut, fanOut, fanIn)
  protected final val bias: ScalarMatrix = if (const != null) const else act.initialize(fanIn, fanOut, fanOut, 1)
  /* Weight-Update Stand-by */
  protected final val dQ: ArrayBuffer[ScalarMatrix] = ArrayBuffer()
  protected final val dL: ScalarMatrix = ScalarMatrix $0(fanOut, fanIn)
  protected final val db = ScalarMatrix $0(fanOut, 1)

  /* Initialization */
  if (quad.nonEmpty) {
    quadratic ++= quad
    quadratic.foreach {
      matx ⇒ dQ += ScalarMatrix.$0(matx.rows, matx.cols)
    }
  } else (0 until fanOut) foreach {
    _ ⇒
      quadratic += act.initialize(fanIn, fanOut, fanInA, fanInB)
      dQ += ScalarMatrix.$0(fanInA, fanInB)
  }

  /**
   * Retrieve first input
   *
   * @param x input to be separated
   * @return first input
   */
  protected def in1(x: ScalarMatrix): ScalarMatrix

  /**
   * Retrive second input
   *
   * @param x input to be separated
   * @return second input
   */
  protected def in2(x: ScalarMatrix): ScalarMatrix

  /**
   * Reconstruct error from fragments
   * @param in1 error of input1
   * @param in2 error of input2
   * @return restored error
   */
  protected def restoreError(in1: ScalarMatrix, in2: ScalarMatrix): ScalarMatrix

  /**
   * Forward computation
   *
   * @param x input matrix
   * @return output matrix
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val inA = in1(x)
    val inB = in2(x)

    val intermediate: ScalarMatrix = linear * x
    intermediate += bias

    val quads = quadratic.map { q ⇒
      val xQ: ScalarMatrix = inA.t * q
      val xQy: ScalarMatrix = xQ * inB
      xQy(0, 0)
    }
    intermediate += ScalarMatrix(quads: _*)

    act(intermediate)
  }

  /**
   * weights for update
   *
   * @return weights
   */
  override val W: IndexedSeq[ScalarMatrix] = bias +: (linear +: quadratic)

  /**
   * accumulated delta values
   *
   * @return delta-weight
   */
  override val dW: IndexedSeq[ScalarMatrix] = db +: (dL +: dQ)

  /**
   * <p>Backward computation.</p>
   *
   * @note <p>
   *       Let this layer have function F composed with function <code> X(x) = x1'.Q.x2 + L.x + b </code>
   *       and higher layer have function G. (Each output is treated as separately except propagation)
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
   * @param error to be propagated ( <code>dG / dF</code> is propagated from higher layer )
   * @return propagated error (in this case, <code>dG/dx</code> )
   */
  protected[deep] override def updateBy(error: ScalarMatrix): ScalarMatrix = {
    val inA = in1(X)
    val inB = in2(X)

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
    db += dGdX

    /*
     * Chain Rule (Linear weight case) : dG/dW_ij = tr[ ( dG/dX ).t * dX/dW_ij ].
     *
     * dX/dW_ij is a fan-Out dimension column vector with all zero but (i, 1) = X_j.
     * Thus, tr(.) can be omitted, and dG/dW_ij = (dX/dW_ij).t * dG/dX
     * Then {j-th column of dG/dW} = X_j * dG/dX = dG/dX * X_j.
     *
     * Therefore dG/dW = dG/dX * X.t
     */
    val dGdL = dGdX * X.t
    dL += dGdL
    /*
     * Chain Rule (Linear weight part) : dG/dx_ij = tr[ ( dG/dX ).t * dX/dx_ij ].
     *
     * X is column vector. Thus j is always 1, so dX/dx_i is a W_?i.
     * Hence dG/dx_i = tr[ (dG/dX).t * dX/dx_ij ] = (W_?i).t * dG/dX.
     *
     * Thus dG/dx (linear part) = W.t * dG/dX.
     */
    val dGdx = linear.t * dGdX

    /*
     * Because X = inA.t * Q * inB, dX/dQ = inA * inB.t
     */
    val dXdQ: ScalarMatrix = inA * inB.t //d tr(axb)/dx = a'b'

    // Add dG/dx quadratic part.
    updateQuadratic(inA, inB, dGdX, dXdQ, dGdx)
  }

  private def updateQuadratic(inA: ScalarMatrix, inB: ScalarMatrix,
                              dGdXAll: ScalarMatrix, dXdQ: ScalarMatrix,
                              acc: ScalarMatrix, id: Int = fanOut - 1): ScalarMatrix =
    if (id >= 0) {
      // This is scalar
      val dGdX = dGdXAll(id, 0)

      /*
       * Chain Rule (Quadratic weight case) : dG/dQ_ij = tr[ ( dG/dX ).t * dX/dQ_ij ].
       *
       * dX/dQ_ij = (inA * inB.t)_ij, and so dG/dQ_ij = (dG/dX).t * dX/dQ_ij.
       * They are scalar, so dG/dQ = dG/dX * dX/dQ.
       */
      val dGdQ: ScalarMatrix = dXdQ :* dGdX
      dQ(id) += dGdQ

      /*
       * Chain Rule (Linear weight part) : dG/dx_ij = tr[ ( dG/dX ).t * dX/dx_ij ].
       *
       * X is column vector. Thus j is always 1, so dX/dx_i is a W_?i.
       * Hence dG/dx_i = tr[ (dG/dX).t * dX/dx_ij ] = (W_?i).t * dG/dX.
       *
       * Thus dG/dx = W.t * dG/dX.
       *
       * Chain Rule (Quadratic weight part) : dG/dx_ij = tr[ ( dG/dX ).t * dX/dx_ij ].
       *
       * Note that x is a column vector with inA, inB as parts.
       * Because X = inA.t * Q * inB, dX/dxA = inB.t * Q.t and dX/dxB = inA.t * Q
       * Since dG/dX is scalar, we obtain dG/dx by scalar multiplication.
       */
      val dXdxQ1: ScalarMatrix = inB.t * quadratic(id).t //d tr(ax')/dx = d tr(x'a)/dx = a'
      val dXdxQ2: ScalarMatrix = inA.t * quadratic(id) //d tr(ax)/dx = d tr(xa)/dx = a
      val dGdx: ScalarMatrix = restoreError(dXdxQ1, dXdxQ2) :* dGdX
      acc += dGdx

      updateQuadratic(inA, inB, dGdXAll, dXdQ, acc, id - 1)
    } else
      acc
}