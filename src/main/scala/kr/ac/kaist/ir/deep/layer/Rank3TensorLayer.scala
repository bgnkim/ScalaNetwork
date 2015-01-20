package kr.ac.kaist.ir.deep.layer

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.fn.act.Activation

/**
 * Layer: Basic, Fully-connected Rank 3 Tensor Layer.
 *
 * <pre>
 * v0 = a column vector concatenate v2 after v1 (v11, v12, ... v1in1, v21, ...)
 * Q = Rank 3 Tensor with size out, in1 × in2 is its entry.
 * L = Rank 3 Tensor with size out, 1 × (in1 + in2) is its entry.
 * b = out × 1 matrix.
 *
 * output = f( v1'.Q.v2 + L.v0 + b )
 * </pre>
 *
 * @param fanIns is the number of input. (vector1, vector2, entire).
 * @param fanOut is the number of output
 * @param act is an activation function to be applied
 * @param quad is initial quadratic-level weight matrix Q for the case that it is restored from JSON (default: Seq())
 * @param lin is initial linear-level weight matrix L for the case that it is restored from JSON (default: Seq())
 * @param const is initial bias weight matrix b for the case that it is restored from JSON (default: null)
 */
abstract class Rank3TensorLayer(protected val fanIns: (Int, Int, Int),
                                protected val fanOut: Int,
                                protected override val act: Activation,
                                quad: Seq[ScalarMatrix] = Seq(),
                                lin: Seq[ScalarMatrix] = Seq(),
                                const: ScalarMatrix = null)
  extends Layer {
  /** Number of Fan-ins */
  protected val fanInA = fanIns._1
  protected val fanInB = fanIns._2
  protected val fanIn = fanIns._3
  /** Initialize weight */
  protected val quadratic: Seq[ScalarMatrix] = if (quad.nonEmpty) quad else (0 until fanOut) map { _ ⇒ act.initialize(fanIn, fanOut, fanInA, fanInB)}
  protected val linear: Seq[ScalarMatrix] = if (lin.nonEmpty) lin else (0 until fanOut) map { _ ⇒ act.initialize(fanIn, fanOut, 1, fanIn)}
  protected val bias: ScalarMatrix = if (const != null) const else act.initialize(fanIn, fanOut, fanOut, 1)
  /** Weight-Update Stand-by */
  protected val dQ = quadratic map { matx ⇒ ScalarMatrix $0(matx.rows, matx.cols)}
  protected val dL = linear map { matx ⇒ ScalarMatrix $0(matx.rows, matx.cols)}
  protected val db = ScalarMatrix $0(fanOut, 1)

  /**
   * Retrieve first input 
   * @param x to be separated
   * @return first input
   */
  protected def in1(x: ScalarMatrix): ScalarMatrix

  /**
   * Retrive second input
   * @param x to be separated
   * @return second input
   */
  protected def in2(x: ScalarMatrix): ScalarMatrix

  /**
   * Forward computation
   * @param x of input matrix
   * @return output matrix
   */
  override def apply(x: ScalarMatrix): ScalarMatrix = {
    val inA = in1(x)
    val inB = in2(x)

    val intermediate = ScalarMatrix $0(fanOut, 1)

    (0 until fanOut).par foreach {
      id ⇒
        val xQ: ScalarMatrix = inA.t * quadratic(id)
        val xQy: ScalarMatrix = xQ * inB
        val Lxy: ScalarMatrix = linear(id) * x
        val X: ScalarMatrix = xQy + Lxy

        // 1 × 1 matrix output.
        intermediate.update(id, 0, X(0, 0) + bias(id, 0))
    }

    act(intermediate)
  }

  /**
   * weights for update
   * @return weights
   */
  override def W: Seq[ScalarMatrix] = bias +: (linear ++ quadratic)

  /**
   * accumulated delta values
   * @return delta-weight
   */
  override def dW: Seq[ScalarMatrix] = db +: (dL ++ dQ)

  /**
   * <p>Backward computation.</p>
   *
   * <p>
   * Let this layer have function F composed with function <code> X(x) = x1'.Q.x2 + L.x + b </code>
   * and higher layer have function G. (Each output is treated as separately except propagation)
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
    val inA = in1(input)
    val inB = in2(input)

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
    val dGdXAll: ScalarMatrix = dFdX * error

    (0 until fanOut).foldLeft(ScalarMatrix $0(fanIn, 1)) {
      (acc, id) ⇒
        // This is scalar
        val dGdX = dGdXAll(id, 0)

        /*
         * Chain Rule (Linear weight case) : dG/dW_ij = tr[ ( dG/dX ).t * dX/dW_ij ].
         *
         * dX/dW_ij is a fan-Out dimension column vector with all zero but (i, 1) = X_j.
         * Thus, tr(.) can be omitted, and dG/dW_ij = (dX/dW_ij).t * dG/dX
         * Then {j-th column of dG/dW} = X_j * dG/dX = dG/dX * X_j.
         *
         * Therefore dG/dW = dG/dX * X.t
         */
        val dGdL: ScalarMatrix = input.t :* dGdX // Because dGdX is scalar, dGdX * X.t = X.t * dGdX
        dL(id) += dGdL

        /*
         * Chain Rule (Quadratic weight case) : dG/dQ_ij = tr[ ( dG/dX ).t * dX/dQ_ij ].
         *
         * Because X = inA.t * Q * inB, dX/dQ = inA * inB.t
         * Therefore dX/dQ_ij = (inA * inB.t)_ij, and so dG/dQ_ij = (dG/dX).t * dX/dQ_ij.
         * They are scalar, so dG/dQ = dG/dX * dX/dQ.
         */
        val dXdQ: ScalarMatrix = inA * inB.t //d tr(axb)/dx = a'b'
        val dGdQ: ScalarMatrix = dXdQ :* dGdX
        dQ(id) += dGdQ

        // For bias, input is always 1. We only need dG/dX
        db(id, 0) += dGdX

        /*
         * Chain Rule (Linear weight part) : dG/dx_ij = tr[ ( dG/dX ).t * dX/dx_ij ].
         *
         * X is column vector. Thus j is always 1, so dX/dx_i is a W_?i.
         * Hence dG/dx_i = tr[ (dG/dX).t * dX/dx_ij ] = (W_?i).t * dG/dX.
         *
         * Thus dG/dx = W.t * dG/dX
         */
        val dGdxL: ScalarMatrix = linear(id).t * dGdX

        /*
         * Chain Rule (Quadratic weight part) : dG/dx_ij = tr[ ( dG/dX ).t * dX/dx_ij ].
         *
         * Note that X is a column vector with inA as upper part, inB as lower part.
         * Because X = inA.t * Q * inB, dX/dxA = inB.t * Q.t and dX/dxB = inA.t * Q
         * Therefore dX/dx is obtained by concatnating them.
         * Since dG/dX is scalar, we obtain dG/dx by scalar multiplication.
         */
        val dXdxQ1: ScalarMatrix = inB.t * quadratic(id).t //d tr(ax')/dx = d tr(x'a)/dx = a'
        val dXdxQ2: ScalarMatrix = inA.t * quadratic(id) //d tr(ax)/dx = d tr(xa)/dx = a
        val dXdxQ: ScalarMatrix = dXdxQ1 col_+ dXdxQ2
        val dGdxQ: ScalarMatrix = dXdxQ :* dGdX

        val dGdx: ScalarMatrix = dGdxL + dGdxQ
        acc += dGdx
    }
  }
}