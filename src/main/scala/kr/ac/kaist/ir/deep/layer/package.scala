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
     * Let this layer have function F composed with function <code> X(x) = w.x + b </code>
     * and higher layer have function G.
     * </p>
     *
     * <p>
     * Weight is updated with: <pre>dG/dw = {dG / dF} * {dF / dX} * {dX / dw}</pre>
     * and propagate <pre>dG/dx = {dG / dF} * {dF / dX} * {dX / dx}</pre>
     * </p>
     *
     * <p>
     * Suppose that error function G : Rn to R1 and input size (with bias) is in Rm. Then dimensions are:
     * <pre>
     * dG / dF : (1, n) matrix
     * dF / dX : (n, n) matrix
     * dX / dx : (n, m) matrix
     * dX / dw : 4th rank tensor with (m, n) matrix with (n, 1) matrix as entries
     *
     * dG / dx : (1, m)
     * dG / dw : (m, n)
     * </pre>
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
     * Weight is updated with: <pre>dG/dW = {dG / dF} * {dF / dX} * {dX / dW}</pre>
     * and propagate <pre>dG/dx = {dG / dF} * {dF / dX} * {dX / dx}</pre>
     * </p>
     *
     * <p>
     * Suppose that error function G : Rn to R1 and input size (with bias) is in Rm. Then dimensions are:
     * <pre>
     * dG / dF : (1, n) matrix
     * dF / dX : (n, n) matrix
     * dX / dx : (n, m) matrix
     * dX / dW : 4th rank tensor with (m, n) matrix with (n, 1) matrix as entries
     * In this case, (j, i) entry of tensor is a column vector of all zeros except i-th element(x_j).
     *
     * dG / dx : (1, m) matrix
     * dG / dW : (m, n) matrix
     * </pre>
     * </p>
     *
     * @param error to be propagated ( <code>dG / dF</code> is propagated from higher layer )
     * @param input of this layer (in this case, <code>x = entry of dX / dw</code>)
     * @param output of this layer (in this case, <code>y</code>)
     * @return propagated error (in this case, <code>dG/dx</code> )
     */
    protected[deep] override def !(error: ScalarMatrix, input: ScalarMatrix, output: ScalarMatrix): ScalarMatrix = {
      // fanOut × fanOut matrix
      val dFdX = act.derivative(output)
      // 1 × fanOut matrix
      val dGdX: ScalarMatrix = error * dFdX

      // CAUTION: differentiation by matrix gives transposed dimension !!!
      // dG/dW is fanIn × fanOut tensor with 1 × 1 matrix (i.e. scalar) as entries.
      // Because (j, i) entry of tensor is a column vector of all zeros except i-th element(x_j),
      // That is, (j, i) entry of dG/dW is (i-th col of {dG / dF} * {dF / dX}) * x_j.
      // Hence j-th row of dG/dW is (dG / dX) * x_j
      // Therefore input * (dG / dX) gives the right thing
      val dGdW: ScalarMatrix = input * dGdX
      delta += dGdW.t

      // For bias, input is always 1. We only need transpose of dG / dX
      dbias += dGdX.t

      // Weight is dX / dx, the fanOut × fanIn matrix.
      val dXdx = weight

      dGdX * dXdx // 1 × fanIn matrix
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
          intermediate.update(id, 1, X(0, 0) + bias(id, 0))
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
     * Weight is updated with: <pre>dG/dW = {dG / dF} * {dF / dX} * {dX / dW}</pre>
     * and propagate <pre>dG/dx = {dG / dF} * {dF / dX} * {dX / dx}</pre>
     * </p>
     *
     * <p>
     * Suppose that error function G : R1 to R1 and input sizes are in Rp1, Rp2. Then dimensions are:
     * <pre>
     * dG / dF : (1, 1) matrix
     * dF / dX : (1, 1) matrix
     * dX / dxi = dQ / dxi + dL / dxi
     * dL / dxi = (1, pi) matrix
     * dQ / dxi : (1, pi) matrix
     * dX / dxi : (1, pi) matrix
     *
     * dX / dW : dX / dQ, dX / dL
     * dX / dL : (p, 1) matrix = x
     * dX / dQ : (p2, p1) matrix = x2.x1'
     *
     * dG / dx : (1, p) matrix
     * dG / dL : (p, 1) matrix
     * dG / dQ : (p2, p1) matrix
     * </pre>
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

      // fanOut × fanOut matrix
      val dFdX = act.derivative(output)
      // 1 × fanOut matrix
      val dGdXAll: ScalarMatrix = error * dFdX

      (0 until fanOut).foldLeft(ScalarMatrix $0(1, fanIn)) {
        (acc, id) ⇒ {
          // This is scalar
          val dGdX = dGdXAll(0, id)

          // For Linear weight: dG/dL is fanIn × 1 tensor with 1 × 1 matrix (i.e. scalar) as entries.
          val dGdL: ScalarMatrix = input :* dGdX // dGdX * input == input * dGdX
          dL(id) += dGdL.t //To update, we need transpose

          // For Quadratic weight: dG/dQ is fanInB × fanInA tensor with 1 × 1 matrix (i.e. scalar) as entries.
          val dXdQ: ScalarMatrix = inA * inB.t //d tr(axb)/dx = a'b'
          val dGdQ: ScalarMatrix = dXdQ :* dGdX
          dQ(id) += dGdQ.t //To update, we need transpose

          // For bias weight: dG/db is 1 × 1. (Which is symmetric)
          db(id, 0) += dGdX

          // By Linear weight: 1 × fanIn matrix
          val dLdx = linear(id) //d tr(ax)/dx = d tr(xa)/dx = a
          // By Quadratic weight: 1 × fanIn matrix
          val dQdx1: ScalarMatrix = inB.t * quadratic(id).t //d tr(ax')/dx = d tr(x'a)/dx = a'
          val dQdx2: ScalarMatrix = inA.t * quadratic(id) //d tr(ax)/dx = d tr(xa)/dx = a
          val dQdx: ScalarMatrix = dQdx1 col_+ dQdx2
          val dXdx: ScalarMatrix = dLdx + dQdx

          acc += dXdx * dGdX // 1 × fanIn matrix
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
      // 1 × Recon matrix
      val dGdX: ScalarMatrix = error * dFdX

      // CAUTION: differentiation by matrix gives transposed dimension !!!
      // dG/dW is Hidden × Recon tensor with 1 × 1 matrix (i.e. scalar) as entries.
      // Because (j, i) entry of tensor is a column vector of all zeros except i-th element(x_j),
      // That is, (j, i) entry of dG/dW is (i-th col of {dG / dF} * {dF / dX}) * x_j.
      // Hence j-th row of dG/dW is (dG / dX) * x_j
      // Therefore input * (dG / dX) gives the right thing
      val dGdW: ScalarMatrix = input * dGdX
      // dGdW has the form of transpose of W.t (Recon × Hidden), we just add it.
      delta += dGdW

      // For bias, input is always 1. We only need transpose of dG / dX
      drBias += dGdX.t

      // Weight is dX / dx, the Recon × Hidden matrix.
      val dXdx = weight.t

      dGdX * dXdx // 1 × Hidden matrix
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
