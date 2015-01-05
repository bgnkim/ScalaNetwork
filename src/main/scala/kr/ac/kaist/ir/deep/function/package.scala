package kr.ac.kaist.ir.deep

/**
 * Package for functions
 *
 * Created by bydelta on 2014-12-27.
 */
package object function {
  /** Type of neuron **/
  type NeuronId = Int
  /** Type of scalar **/
  type Scalar = Double
  /** Type of Neuron Input **/
  type NeuronVector = Map[NeuronId, Scalar]

  private val subtract = (x: Scalar, y: Scalar) ⇒ x - y
  private val addition = (x: Scalar, y: Scalar) ⇒ x + y
  private val multiply = (x: Scalar, y: Scalar) ⇒ x * y

  /**
   * Operation for NeuronVector
   * @param seq is left hand side for each expression.
   */
  implicit class NeuronVectorOp(val seq: NeuronVector) extends AnyVal {

    /**
     * Subtract operation
     * @param that is right hand side of subtraction.
     * @return {seq - that}, defined on at least one of them (If undefined, then it is treated as zero.)
     */
    def -(that: NeuronVector) = (seq, that) apply subtract

    /**
     * Addition operation
     * @param that is right hand side of addition.
     * @return {seq + that}, defined on at least one of them (If undefined, then it is treated as zero.)
     */
    def +(that: NeuronVector) = (seq, that) apply addition

    /**
     * Element-wise Multiplication operation
     * @param that is right hand side of multiplication.
     * @return {seq + that}, defined on at least one of them (If undefined, then it is treated as zero.)
     */
    def *(that: NeuronVector) = (seq, that) apply multiply

    /**
     * Scalar Multiplication
     * @param that is multiplier
     * @return {seq * that}, defined on domain of seq
     */
    def *(that: Scalar) = apply(x ⇒ x * that)

    def dot(that: NeuronVector) = (seq * that).$sum

    /**
     * Sum-up all entries of the given vector
     * @return sum of entries.
     */
    def $sum = seq.foldLeft(0.0)((sum, tpl) ⇒ sum + tpl._2)

    /**
     * Power operation
     * @param power is the power
     * @return {seq ** power}.
     */
    def **(power: Scalar) = apply(Math.pow(_, power))

    /**
     * Compute vector applied by given function.
     * @param fn to be applied
     * @return {fn(seq)}
     */
    def apply(fn: Scalar ⇒ Scalar) = seq mapValues fn
  }

  /**
   * Operation for NeuronVector
   * @param set is (left, right) hand side for each expression.
   */
  implicit class NeuronVectorOp2(val set: (NeuronVector, NeuronVector)) extends AnyVal {
    /**
     * Apply given function as a binary operation.
     * @param fn for applied
     * @return fn({first}, {second})
     */
    def apply(fn: (Scalar, Scalar) ⇒ Scalar) = {
      val set1 = set._1
      val set2 = set._2
      val keys = set._1.keySet union set._2.keySet
      (keys map {
        id ⇒ id → fn(set1.getOrElse(id, 0.0), set2.getOrElse(id, 0.0))
      }).toMap
    }
  }

  /** Define Alias **/
  val Tanh = HyperbolicTangent

  /**
   * Define Activation Function trait.
   */
  trait Activation extends (Scalar ⇒ Scalar) with Serializable {
    /**
     * Compute derivative of this function
     * @param fx is output of this function
     * @return differentiation value at f(x)=fx.
     */
    def derivative(fx: Scalar): Scalar

    /**
     * Compute mapping at x
     * @param x is input scalar.
     * @return f(x)
     */
    def apply(x: Scalar): Scalar

    /**
     * Initialize Weight Vector
     * @param fanIn is a weight vector indicates fan-in
     * @param fanOut is a count of fan-out
     * @return random number generator
     */
    def initialize(fanIn: Int, fanOut: Int) = Math.random _
  }

  /**
   * Define Objective Function trait.
   */
  trait Objective extends ((NeuronVector, NeuronVector) ⇒ Scalar) with Serializable {
    /**
     * Compute derivative of this objective function
     * @param real is expected real output
     * @param output is computational output of the network
     * @return is differentiation(Gradient) vector at f(X)=output, i.e. error of each output neuron.
     */
    def derivative(real: NeuronVector, output: NeuronVector): NeuronVector

    /**
     * Compute error
     * @param real is expected real output
     * @param output is computational output of the network
     * @return is the error
     */
    override def apply(real: NeuronVector, output: NeuronVector): Scalar
  }

  /**
   * Activation Function: Sigmoid
   */
  object Sigmoid extends Activation {
    /**
     * Compute derivative of this function
     * @param fx is output of this function
     * @return differentiation value at f(x)=fx.
     */
    override def derivative(fx: Scalar): Scalar = fx * (1.0 - fx)

    /**
     * Compute mapping at x
     * @param x is input scalar.
     * @return f(x)
     */
    override def apply(x: Scalar): Scalar = 1.0 / (1.0 + Math.exp(-x))

    /**
     * Initialize Weight Vector
     * @param fanIn is a weight vector indicates fan-in
     * @param fanOut is a count of fan-out
     * @return random number generator
     */
    override def initialize(fanIn: Int, fanOut: Int) = {
      val range = Math.sqrt(6.0 / (fanIn + fanOut)) * 4.0
      () ⇒ (Math.random() * 2.0 - 1.0) * range
    }
  }

  /**
   * Activation Function: Tanh
   */
  object HyperbolicTangent extends Activation {
    /**
     * Compute derivative of this function
     * @param fx is output of this function
     * @return differentiation value at f(x)=fx.
     */
    override def derivative(fx: Scalar): Scalar = 1.0 - fx * fx

    /**
     * Compute mapping at x
     * @param x is input scalar.
     * @return f(x)
     */
    override def apply(x: Scalar): Scalar = Math.tanh(x)

    /**
     * Initialize Weight Vector
     * @param fanIn is a weight vector indicates fan-in
     * @param fanOut is a count of fan-out
     * @return random number generator
     */
    override def initialize(fanIn: Int, fanOut: Int) = {
      val range = Math.sqrt(6.0 / (fanIn + fanOut))
      () ⇒ (Math.random() * 2.0 - 1.0) * range
    }
  }

  /**
   * Activation Function: Rectifier
   *
   * Rectifier(x) = x if x > 0, otherwise 0
   */
  object Rectifier extends Activation {
    /**
     * Compute derivative of this function
     * @param fx is output of this function
     * @return differentiation value at f(x)=fx.
     */
    override def derivative(fx: Scalar): Scalar = if (fx > 0) 1.0 else 0.0

    /**
     * Compute mapping at x
     * @param x is input scalar.
     * @return f(x)
     */
    override def apply(x: Scalar): Scalar = if (x > 0) x else 0.0
  }

  /**
   * Activation Function: Softplus
   *
   * Softplus(x) = log(1 + e ** x)
   */
  object Softplus extends Activation {
    /**
     * Compute derivative of this function
     * @param fx is output of this function
     * @return differentiation value at f(x)=fx.
     */
    override def derivative(fx: Scalar): Scalar = (Math.exp(fx) - 1.0) / Math.exp(fx)

    /**
     * Compute mapping at x
     * @param x is input scalar.
     * @return f(x)
     */
    override def apply(x: Scalar): Scalar = Math.log(1.0 + Math.exp(x))
  }

  /**
   * Objective Function: Sum of Square Error
   */
  object SquaredErr extends Objective {
    /**
     * Compute derivative of this objective function
     * @param real is expected real output
     * @param output is computational output of the network
     * @return is differentiation(Gradient) vector at f(X)=output, i.e. error of each output neuron.
     */
    override def derivative(real: NeuronVector, output: NeuronVector): NeuronVector = output - real

    /**
     * Compute error
     * @param real is expected real output
     * @param output is computational output of the network
     * @return is the error
     */
    override def apply(real: NeuronVector, output: NeuronVector): Scalar = 0.5 * ((real - output) ** 2).$sum
  }

  /**
   * Objective Function: Sum of Cross-Entropy (Logistic)
   */
  object CrossEntropyErr extends Objective {
    /**
     * Entropy function
     */
    val entropy = (r: Scalar, o: Scalar) ⇒ (if (o < 1) -(1.0 - r) * Math.log(1.0 - o) else 0.0) + (if (o > 0) -r * Math.log(o) else 0.0)

    /**
     * Derivative of Entropy function
     */
    val entropyDiff = (r: Scalar, o: Scalar) ⇒ -(r - o) / (o * (1.0 - o))

    /**
     * Compute derivative of this objective function
     * @param real is expected real output
     * @param output is computational output of the network
     * @return is differentiation(Gradient) vector at f(X)=output, i.e. error of each output neuron.
     */
    override def derivative(real: NeuronVector, output: NeuronVector): NeuronVector = (real, output) apply entropyDiff

    /**
     * Compute error
     * @param real is expected real output
     * @param output is computational output of the network
     * @return is the error
     */
    override def apply(real: NeuronVector, output: NeuronVector): Scalar = ((real, output) apply entropy).$sum
  }

  /** Noise Generator Type **/
  type Noise = () ⇒ Double

  /**
   * Noise: Gaussian
   */
  object GaussianNoise extends ((Double, Double) ⇒ Double) with Noise {

    /**
     * Return Gaussian Random Number with mean 0, standard deviation 1.
     * @return random noise
     */
    override def apply() = {
      val u1 = Math.random()
      val u2 = Math.random()

      Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
    }

    /**
     * Return Gaussian Random Number with given specification
     * @param mean of distribution
     * @param stdev of distribution
     * @return random noise
     */
    override def apply(mean: Double, stdev: Double): Double = apply() * stdev + mean
  }

  /**
   * Noise: Uniform
   */
  object UniformNoise extends Noise {
    /**
     * Return uniform random noise.
     * @return random noise.
     */
    override def apply() = Math.random()
  }

}
