package kr.ac.kaist.ir.deep

import kr.ac.kaist.ir.deep.function._
import play.api.libs.json.{JsArray, JsObject, JsValue, Json}

/**
 * Package for layer implementation
 *
 * Created by bydelta on 2015-01-06.
 */
package object layer {

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

  /**
   * Trait of Layer : Reconstructable
   */
  trait Reconstructable extends Layer {
    /**
     * Sugar: Forward computation + reconstruction
     *
     * @param x of hidden layer output matrix
     * @return tuple of reconstruction output
     */
    def rec_>>:(x: ScalarMatrix): ScalarMatrix

    /**
     * Backpropagation of reconstruction. For the information about backpropagation calculation, see [[Layer.!(error, input, output)]]
     * @param error to be propagated 
     * @param input of this layer
     * @param output is final reconstruction output of this layer
     * @return propagated error
     */
    protected[deep] def rec_!(error: ScalarMatrix, input: ScalarMatrix, output: ScalarMatrix): ScalarMatrix
  }

  /**
   * Layer: Basic, Fully-connected Layer
   * @param IO is a pair of input & output, such as 2 -> 3
   * @param act is an activation function to be applied
   * @param w is initial weight matrix for the case that it is restored from JSON
   * @param b is inital bias matrix for the case that it is restored from JSON
   */
  class BasicLayer(IO: (Int, Int), protected override val act: Activation, w: ScalarMatrix = null, b: ScalarMatrix = null) extends Layer {
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
  }

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
   * @param IO is a pair of input & output, such as (2, 2) -> 3
   * @param act is an activation function to be applied
   * @param quad is initial quadratic-level weight matrix Q for the case that it is restored from JSON
   * @param lin is initial linear-level weight matrix L for the case that it is restored from JSON
   * @param const is initial bias weight matrix b for the case that it is restored from JSON
   */
  class Rank3TensorLayer(IO: ((Int, Int), Int), protected override val act: Activation, quad: Seq[ScalarMatrix] = Seq(), lin: Seq[ScalarMatrix] = Seq(), const: ScalarMatrix = null) extends Layer {
    /** Number of Fan-ins */
    protected val fanInA = IO._1._1
    protected val fanInB = IO._1._2
    protected val fanIn = fanInA + fanInB
    /** Number of Fan-outs */
    protected val fanOut = IO._2
    /** Initialize weight */
    protected val quadratic: Seq[ScalarMatrix] = if (quad.nonEmpty) quad else (0 until fanOut) map { _ ⇒ act.initialize(fanIn, fanOut, fanInA, fanInB)}
    protected val linear: Seq[ScalarMatrix] = if (lin.nonEmpty) lin else (0 until fanOut) map { _ ⇒ act.initialize(fanIn, fanOut, 1, fanIn)}
    protected val bias: ScalarMatrix = if (const != null) const else act.initialize(fanIn, fanOut, fanOut, 1)
    /** Weight-Update Stand-by */
    protected val dQ = quadratic map { matx ⇒ ScalarMatrix $0(matx.rows, matx.cols)}
    protected val dL = linear map { matx ⇒ ScalarMatrix $0(matx.rows, matx.cols)}
    protected val db = ScalarMatrix $0(fanOut, 1)

    /**
     * Forward computation
     * @param x of input matrix
     * @return output matrix
     */
    override def apply(x: ScalarMatrix): ScalarMatrix = {
      val inA = x(0 until fanInA, ::)
      val inB = x(fanInA to -1, ::)

      val intermediate = ScalarMatrix $0(fanOut, 1)

      (0 until fanOut) map {
        id ⇒ {
          val xQ: ScalarMatrix = inA.t * quadratic(id)
          val xQy: ScalarMatrix = xQ * inB
          val Lxy: ScalarMatrix = linear(id) * x
          val X: ScalarMatrix = xQy + Lxy

          // 1 × 1 matrix output.
          intermediate.update(id, 0, X(0, 0) + bias(id, 0))
        }
      }

      act(intermediate)
    }

    /**
     * Translate this layer into JSON object (in Play! framework)
     * @return JSON object describes this layer
     */
    override def toJSON: JsObject = Json.obj(
      "type" → "Rank3TensorLayer",
      "in" → Json.arr(fanInA, fanInB),
      "out" → fanOut,
      "act" → act.getClass.getSimpleName,
      "quadratic" → JsArray.apply(quadratic.map(_.to2DSeq)),
      "linear" → JsArray.apply(linear.map(_.to2DSeq)),
      "bias" → bias.to2DSeq
    )

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
      val inA = input(0 until fanInA, ::)
      val inB: ScalarMatrix = input(fanInA to -1, ::)

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
        (acc, id) ⇒ {
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
  }

  /**
   * Layer : Reconstructable Basic Layer
   * @param IO is a pair of input & output, such as 2 -> 3
   * @param act is an activation function to be applied
   * @param w is initial weight matrix for the case that it is restored from JSON
   * @param b is inital bias matrix for the case that it is restored from JSON
   * @param rb is initial reconstruct bias matrix for the case that it is restored from JSON
   */
  class ReconBasicLayer(IO: (Int, Int), act: Activation, w: ScalarMatrix = null, b: ScalarMatrix = null, rb: ScalarMatrix = null) extends BasicLayer(IO, act, w, b) with Reconstructable {
    protected val reBias = if (rb != null) rb else act initialize(fanIn, fanOut, fanIn, 1)
    protected val drBias = ScalarMatrix $0(fanIn, 1)

    /**
     * Sugar: Forward computation + reconstruction
     *
     * @param x of hidden layer output matrix
     * @return tuple of reconstruction output
     */
    override def rec_>>:(x: ScalarMatrix): ScalarMatrix = {
      val wx: ScalarMatrix = weight.t[ScalarMatrix, ScalarMatrix] * x
      val wxb: ScalarMatrix = wx + reBias
      act(wxb)
    }

    /**
     * Backpropagation of reconstruction. For the information about backpropagation calculation, see [[Layer.!(error, input, output)]]
     * @param error to be propagated 
     * @param input of this layer
     * @param output is final reconstruction output of this layer
     * @return propagated error
     */
    protected[deep] override def rec_!(error: ScalarMatrix, input: ScalarMatrix, output: ScalarMatrix): ScalarMatrix = {
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

    /**
     * weights for update
     * @return weights
     */
    override def W: Seq[ScalarMatrix] = reBias +: super.W

    /**
     * accumulated delta values
     * @return delta-weight
     */
    override def dW: Seq[ScalarMatrix] = drBias +: super.dW

    /**
     * Translate this layer into JSON object (in Play! framework)
     * @return JSON object describes this layer
     */
    override def toJSON: JsObject = super.toJSON + ("reconst_bias" → reBias.to2DSeq)
  }

  // TODO class ReconRank3TensorLayer


  // TODO class WeightDepBasicLayer
  // TODO class WeightDepRank3TensorLayer

  /**
   * Companion object of Layer
   */
  object Layer {
    /** Sequence of supported activation functions */
    private val acts = Seq(Sigmoid, HyperbolicTangent, Rectifier, Softplus)

    /**
     * Load layer from JsObject
     * @param obj to be parsed
     * @return New layer reconstructed from this object
     */
    def apply(obj: JsValue) = {
      val in = obj \ "in"
      val out = (obj \ "out").as[Int]

      val actStr = (obj \ "act").as[String]
      val act = (acts find {
        x ⇒ x.getClass.getSimpleName == actStr
      }).getOrElse(HyperbolicTangent)

      val b = ScalarMatrix restore (obj \ "bias").as[Seq[Seq[Scalar]]]

      (obj \ "type").as[String] match {
        case "BasicLayer" ⇒
          val w = ScalarMatrix restore (obj \ "weight").as[Seq[Seq[Scalar]]]
          (obj \ "reconst_bias").asOpt[Seq[Seq[Scalar]]] match {
            case Some(rb) ⇒
              new ReconBasicLayer(in.as[Int] → out, act, w, b, ScalarMatrix restore rb)
            case None ⇒
              new BasicLayer(in.as[Int] → out, act, w, b)
          }
        case "Rank3TensorLayer" ⇒
          val tuple = in.as[Seq[Int]]
          val quad = (obj \ "quadratic").as[Seq[Seq[Seq[Scalar]]]] map { x ⇒ ScalarMatrix restore x}
          val linear = (obj \ "linear").as[Seq[Seq[Seq[Scalar]]]] map { x ⇒ ScalarMatrix restore x}
          new Rank3TensorLayer((tuple(0), tuple(1)) → out, act, quad, linear, b)
      }
    }
  }

}
